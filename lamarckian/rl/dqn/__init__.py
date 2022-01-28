"""
Copyright (C) 2020

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import types
import collections
import contextlib
import random
import hashlib
import pickle
import threading
import queue
import traceback
import asyncio

import numpy as np
import torch
import torch.nn.functional as F
import humanfriendly
import glom
import deepmerge
import recordtype

import lamarckian
from lamarckian.mdp import rollout
from .. import RL1, Remote
from . import agent

NAME = os.path.basename(os.path.dirname(__file__))
Outcome = recordtype.recordtype('Outcome', ['results', 'loss'])


class Buffer(object):
    def __init__(self, receive, capacity, batch_size, device, **kwargs):
        assert batch_size <= capacity, (batch_size, capacity)
        self.receive = receive
        self.transitions = collections.deque(maxlen=capacity)
        self.batch_size = batch_size
        self.device = device
        self.results = []
        self.queue = queue.Queue(maxsize=1)
        while len(self) < self.batch_size:
            self.fill()
        self.random = random.Random(glom.glom(kwargs['config'], 'seed', default=None))
        self.running = True
        self.thread = dict(
            receiving=threading.Thread(target=self.receiving),
            prefetching=threading.Thread(target=self.prefetching)
        )
        for thread in self.thread.values():
            thread.start()

    def close(self):
        self.running = False
        for thread in self.thread.values():
            thread.join()

    def fill(self):
        transitions, results = self.receive()
        self.transitions += [dict(cost=cost, tensors=tensors) for cost, tensors in transitions]
        self.results += results

    def __len__(self):
        return len(self.transitions)

    def receiving(self):
        try:
            while self.running:
                self.fill()
        except Exception as e:
            traceback.print_exc()

    def prefetching(self):
        try:
            while self.running:
                batch = self.random.sample(self.transitions, self.batch_size)
                cost = sum(item.pop('cost', 0) for item in batch)
                tensors = tuple(lamarckian.rl.cat(t) for t in zip(*(item['tensors'] for item in batch)))
                tensors = lamarckian.rl.to_device(self.device, *tensors)
                self.queue.put((cost, tensors))
        except Exception as e:
            traceback.print_exc()
            self.queue.put(e)

    def __next__(self):
        result = self.queue.get()
        if isinstance(result, Exception):
            raise result
        else:
            cost, tensors = result
            results, self.results = self.results, []
            return cost, tensors, results


@lamarckian.util.rpc.wrap.all
@lamarckian.util.rpc.wrap.any
@lamarckian.util.rpc.wrap.gather
class _Actor(RL1):
    pass


class Actor(_Actor):
    Tensors = collections.namedtuple('Tensors', ['inputs', 'inputs_', 'action', 'reward', 'discount', 'legal'])

    def __init__(self, *args, **kwargs):
        torch.set_num_threads(1)
        super().__init__(*args, **kwargs)
        encoding = self.describe()['blob']
        discount = glom.glom(kwargs['config'], 'rl.discount')
        self.hparam.setup('discount', discount, float)
        for name in encoding['reward']:
            self.hparam.setup(f"discount_{name}", glom.glom(kwargs['config'], f"rl.discount_{name}", default=discount), np.float)
        self.hparam.setup('epsilon', glom.glom(kwargs['config'], f"rl.{NAME}.epsilon"), np.float)
        self.epsilon_next = eval('lambda epsilon: ' + glom.glom(kwargs['config'], f"rl.{NAME}.epsilon_next", default='epsilon'))

    def describe_model(self):
        return glom.glom(self.kwargs['config'], 'model.backbone') + glom.glom(self.kwargs['config'], 'model.wrap', default=[])

    def describe_agent(self):
        return dict(
            train=[f"{agent.__name__}.Train"] + glom.glom(self.kwargs['config'], 'rl.agent.wrap', default=[]),
            eval=[f"{agent.__name__}.Eval"] + glom.glom(self.kwargs['config'], 'rl.agent.wrap', default=[]),
        )

    def describe(self):
        encoding = deepmerge.always_merger.merge(super().describe(), self.mdp.describe())
        for model in encoding['blob']['models']:
            model['cls'] = self.describe_model()
        encoding['blob']['agent'] = self.describe_agent()
        for coding, boundary in self.hparam.describe().items():
            encoding[coding] = boundary
        return encoding

    def rollout(self):
        me, opponent = self.spawn()
        loop = asyncio.get_event_loop()
        battle = self.mdp.reset(me, *opponent, loop=loop)
        with contextlib.closing(battle), contextlib.closing(self.agent.train(self.model, hparam=self.hparam, **self.kwargs)) as agent, contextlib.closing(self.make_agents(opponent)) as agents:
            trajectory, exp = loop.run_until_complete(asyncio.gather(
                rollout.get_trajectory(battle.controllers[0], agent),
                *[rollout.get_cost(controller, agent) for controller, agent in zip(battle.controllers[1:], agents.values())],
                *battle.ticks,
            ))[0]
            result = battle.controllers[0].get_result()
            if opponent:
                result['digest_opponent_train'] = hashlib.md5(pickle.dumps(list(opponent.values()))).hexdigest()
            return trajectory, exp, [result]

    def to_transition(self, trajectory, exp):
        inputs = [tuple(t.cpu() for t in exp['inputs']) for exp in trajectory + [exp]]
        action      = torch.LongTensor(np.stack([exp['action'] for exp in trajectory]))
        reward      = torch.FloatTensor(np.stack([exp['reward'] for exp in trajectory]))
        discount    = torch.FloatTensor(np.array([not exp['done'] for exp in trajectory])).unsqueeze(-1) * self.hparam['discount']
        if 'legal' in exp.keys():
            legal = torch.cat([exp['legal'] for exp in trajectory])
        else:
            legal = torch.ones((len(trajectory), 1), dtype=int)
        return [(exp.get('cost', 1), self.Tensors(*tensors)) for exp, tensors in zip(trajectory, zip(inputs[:-1], inputs[1:], action, reward, discount, legal))]

    def __getstate__(self):
        state = super().__getstate__()
        state['mdp'] = self.mdp.__getstate__()
        return state

    def gather(self):
        with torch.no_grad():
            trajectory, exp, results = self.rollout()
            transitions = self.to_transition(trajectory, exp)
            self.hparam['epsilon'] = self.epsilon_next(self.hparam['epsilon'])
            return transitions, results


class Learner(Remote):
    Actor = Actor
    Tensors = Actor.Tensors

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = getattr(F, glom.glom(kwargs['config'], f"rl.{NAME}.loss", default='mse_loss'))
        self.rpc_broadcast = lamarckian.util.rpc.All(self.actors, **kwargs)
        self.rpc_gather = lamarckian.util.rpc.Gather(self.rpc_all, **kwargs)
        self.hparam.setup('batch_size', glom.glom(kwargs['config'], 'train.batch_size'), np.int)
        self.hparam.setup('lr', glom.glom(kwargs['config'], 'train.lr'), np.float)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        encoding = self.describe()
        module = lamarckian.evaluator.parse(*encoding['blob']['models'][self.me]['cls'], **kwargs)
        model_args = encoding['blob']['models'][self.me]
        self.model_target = module(**model_args)        
        self.model_target.to(self.device)
        self.model_target.eval()

    def close(self):
        self.rpc_broadcast.close()
        self.rpc_gather.close()
        return super().close()

    def training(self):
        self.should_update = lamarckian.util.counter.Number(humanfriendly.parse_size(str(glom.glom(self.kwargs['config'], f"rl.{NAME}.update"))))
        self.model_target.load_state_dict(self.model.state_dict())
        training = super().training()
        self.broadcaster = lamarckian.rl.remote.Runner(lambda: self.sync_blob(self.get_blob()))
        gathering = self.rpc_gather.gathering('gather')

        capacity = str(glom.glom(self.kwargs['config'], f"rl.{NAME}.capacity"))
        self.buffer = Buffer(
            self.rpc_gather, 
            humanfriendly.parse_size(capacity), 
            self.hparam['batch_size'], 
            self.device,
            **self.kwargs
        )

        def close():
            self.buffer.close()
            gathering.close()
            self.broadcaster.close()
            training.close()
        return types.SimpleNamespace(close=close)

    def __next__(self):
        cost, tensors, results = next(self.buffer)
        return cost, self.Tensors(*tensors), results

    def get_action(self, *inputs):
        q, = self.model(*inputs)[:1]
        _, action = q.max(1)
        return action

    def get_value(self, inputs, action):
        with torch.no_grad():
            q_target = self.model_target(*inputs)['discrete'][0]
        return q_target.gather(1, action.view(-1, 1)).view(-1)

    def __call__(self):
        cost, tensors, results = next(self)
        self.cost += cost
        self.optimizer.zero_grad()
        q = self.model(*tensors.inputs)['discrete'][0]
        credit = tensors.reward + tensors.discount * self.get_value(tensors.inputs_, action=q.max(1)[1])
        loss = self.loss(q.gather(1, tensors.action.view(-1, 1)).view(-1), credit.detach())
        loss.backward()
        self.optimizer.step()
        self.broadcaster()
        if self.should_update():
            self.model_target.load_state_dict(self.model.state_dict())
        return Outcome(results, loss)


class Evaluator(Learner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recorder = lamarckian.util.recorder.Recorder.new(**kwargs)
        self.saver = lamarckian.evaluator.Saver(self)
        self.profiler = lamarckian.evaluator.record.Profiler(self.cost, len(self), **kwargs)
        self.recorder.register(
            lamarckian.util.counter.Time(**glom.glom(kwargs['config'], 'record.save')),
            lambda *args, **kwargs: self.saver(),
        )
        self.recorder.register(
            lamarckian.util.counter.Time(**glom.glom(kwargs['config'], 'record.scalar')),
            lambda *args, **kwargs: self.profiler(self.cost),
        )
        self.recorder.register(
            lamarckian.util.counter.Time(**glom.glom(kwargs['config'], 'record.scalar')),
            lambda *args, **kwargs: lamarckian.util.record.Scalar(self.cost, **lamarckian.util.duration.stats),
        )
        self.recorder.register(
            lamarckian.rl.record.Rollout.counter(**kwargs),
            lambda *args, **kwargs: lamarckian.rl.record.Rollout.new(**kwargs)(self.cost, self.get_blob()),
        )
        self.recorder.register(
            lamarckian.util.counter.Time(**glom.glom(kwargs['config'], 'record.scalar')),
            lambda *args, **kwargs: lamarckian.util.record.Scalar(self.cost, **self.results()),
        )
        self.recorder.register(
            lamarckian.util.counter.Time(**glom.glom(kwargs['config'], 'record.scalar')),
            lambda outcome: lamarckian.util.record.Scalar(self.cost, **{f"{NAME}/loss": outcome['loss'].item()}),
        )
        self.recorder.register(
            lamarckian.util.counter.Time(**glom.glom(kwargs['config'], 'record.model')),
            lambda outcome: lamarckian.rl.record.Model(f"{NAME}/model", self.cost, self.get_blob()),
        )
        self.recorder.put(lamarckian.rl.record.Graph(f"{NAME}/graph", self.cost))
        self.recorder.put(lamarckian.util.record.Text(self.cost, topology=repr(self.rpc_all)))

    def close(self):
        self.saver()
        self.recorder.close()
        return super().close()

    def training(self):
        self.results = lamarckian.rl.record.Results(**self.kwargs)
        return super().training()

    def __call__(self):
        outcome = super().__call__()
        self.results += outcome.results
        self.recorder(outcome)
        return outcome
