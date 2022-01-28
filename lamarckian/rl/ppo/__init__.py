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
import math
import contextlib
import types
import pickle
import hashlib
import threading
import queue
import traceback
import asyncio

import numpy as np
import torch
import torch.nn.functional as F
import glom

import lamarckian
from lamarckian.mdp import rollout
from .. import RL, Remote, ac, wrap as _wrap

NAME = os.path.basename(os.path.dirname(__file__))

from . import wrap, record


class Prefetcher(object):
    def __init__(self, receive, batch_size, dim=0, capacity=1):
        self.receive = receive
        self.batch_size = batch_size
        self.dim = dim
        self.results = []
        if capacity > 0:
            self.queue = queue.Queue(capacity)
        else:
            self.cost = 0
            tensors = lamarckian.rl.make_batch(self.receive_tensor, batch_size, dim)

            def get(*args, **kwargs):
                cost, tensors = self.queue.data
                self.queue.data = (0, tensors)
                return cost, tensors
            self.queue = types.SimpleNamespace(
                data=(self.cost, tensors),
                put=lambda data, *args, **kwargs: setattr(self.queue, 'data', data),
                get=get,
            )
        self.running = True
        self.thread = threading.Thread(target=self.receiving)
        self.thread.start()

    def close(self):
        self.running = False
        thread = threading.Thread(target=self.thread.join)
        thread.start()
        while thread.is_alive():
            try:
                self.queue.get(block=False)
            except queue.Empty:
                pass
        thread.join()

    def receive_tensor(self):
        cost, tensors, results, iteration = self.receive()
        self.cost += cost
        self.results += results
        return tensors

    def receiving(self):
        try:
            while self.running:
                self.cost = 0
                tensors = lamarckian.rl.make_batch(self.receive_tensor, self.batch_size, self.dim)
                self.queue.put((self.cost, tensors))
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


class Truncator(lamarckian.rl.Truncator):
    def __init__(self, rl, step, cast=rollout.cast):
        super().__init__(rl, step, cast=cast)
        assert 0 < step < np.iinfo(np.int).max, step
        self.step = step

    def __next__(self):
        trajectory = []
        results = []
        while len(trajectory) < self.step:
            if self.done:
                self._reset()
            task = asyncio.Task(rollout.get_trajectory(self.battle.controllers[0], self.agent, step=self.step - len(trajectory), cast=self.cast), loop=self.loop)
            lamarckian.mdp.util.wait(task, self.tasks, self.loop)
            trajectory1, exp = task.result()
            trajectory += trajectory1
            self.done = trajectory[-1]['done']
            if self.done:
                result = self.battle.controllers[0].get_result()
                if self.opponent:
                    result['digest_opponent_train'] = hashlib.md5(pickle.dumps(list(self.opponent.values()))).hexdigest()
                results.append(result)
                # self.loop.run_until_complete(asyncio.gather(*self.tasks))
        assert len(trajectory) == self.step, (len(trajectory), self.step)
        return trajectory, exp, results


class Actor(lamarckian.util.rpc.wrap.gather(lamarckian.util.rpc.wrap.any(lamarckian.util.rpc.wrap.all(ac.Actor)))):
    def __init__(self, *args, **kwargs):
        torch.set_num_threads(1)
        super().__init__(*args, **kwargs)

    def create_truncator(self):
        return Truncator(self, self.hparam['truncation'])

    def rollout(self):
        return next(self.truncator)

    def set_blob_(self, blob, iteration):
        self.set_blob(blob)
        self.iteration = iteration

    def to_tensor(self, trajectory, exp):
        tensors = super().to_tensor(trajectory, exp)
        tensors = {key: [t.unsqueeze(1) for t in value] if isinstance(value, (tuple, list)) else value.unsqueeze(1) for key, value in tensors.items()}
        tensors['inputs'] = tuple(t.unsqueeze(1) for t in map(torch.cat, zip(*[exp['inputs'] for exp in trajectory + [exp]])))
        return tensors

    def gather(self):
        with torch.no_grad():
            trajectory, exp, results = self.rollout()
            cost = sum(exp.get('cost', 1) for exp in trajectory)
            tensors = self.to_tensor(trajectory, exp)
            return cost, tensors, results, self.iteration


class Learner(Remote):
    Actor = Actor
    DIM = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparam.setup('lr', glom.glom(kwargs['config'], 'train.lr'), np.float)
        for key in 'policy critic entropy'.split():
            self.hparam.setup(f'weight_loss_{key}', glom.glom(kwargs['config'], f"rl.ac.weight_loss.{key}"), np.float)
        self.loss_critic = getattr(F, glom.glom(kwargs['config'], 'rl.ac.loss_critic', default='mse_loss'))
        self.rpc_broadcast = lamarckian.util.rpc.All(self.actors, **kwargs)
        self.rpc_gather = lamarckian.util.rpc.Gather(self.rpc_all, **kwargs)
        self.iteration = 0
        self.hparam.setup('batch_size', glom.glom(kwargs['config'], 'train.batch_size'), np.int)
        self.hparam.setup('clip', glom.glom(kwargs['config'], f'rl.{NAME}.clip', default=np.finfo(np.float32).max), np.float)
        self.hparam.setup('reuse', glom.glom(kwargs['config'], f'rl.{NAME}.reuse', default=1), np.int)
        self.norm_advantage = eval('lambda advantage: ' + glom.glom(kwargs['config'], f'rl.ac.norm_advantage', default='advantage'))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def close(self):
        self.rpc_broadcast.close()
        self.rpc_gather.close()
        return super().close()

    def sync_blob(self, blob):
        return self.rpc_broadcast('set_blob_', blob, self.iteration)

    def receive(self):
        cost, tensors, results, iteration = self.rpc_gather()
        tensors = lamarckian.rl.to_device(self.device, **tensors)
        return cost, tensors, results, iteration

    def training(self):
        training = super().training()
        self.broadcaster = lamarckian.rl.remote.Runner(lambda: self.sync_blob(self.get_blob()))
        gathering = self.rpc_gather.gathering('gather')
        if self.DIM:
            batch_size = math.ceil(self.hparam['batch_size'] / self.hparam['truncation'])
        else:
            batch_size = self.hparam['batch_size']
        self.prefetcher = Prefetcher(self.receive, batch_size, self.DIM, glom.glom(self.kwargs['config'], f"rl.{NAME}.prefetch", default=1))
        encoding = self.describe()['blob']
        self.discount = torch.FloatTensor([self.hparam[f"discount_{name}"] for name in encoding['reward']]).to(self.device)
        self.gae = torch.FloatTensor([self.hparam[f"gae_{name}"] for name in encoding['reward']]).to(self.device)

        def close():
            self.prefetcher.close()
            gathering.close()
            self.broadcaster.close()
            training.close()
        return types.SimpleNamespace(close=close)

    def forward(self, old):
        inputs = tuple(input.view(-1, *input.shape[2:]) for input in old['inputs'])
        outputs = self.model(*inputs)
        shape = old['inputs'][0].shape[:2]
        outputs = {key: value.view(*shape, *value.shape[1:]) if torch.is_tensor(value) else [t.view(*shape, *t.shape[1:]) for t in value] for key, value in outputs.items()}
        values = outputs['critic']
        outputs['critic'], outputs['critic_'] = values[:-1], values[1:]
        logp = []
        entropy = []
        with contextlib.closing(self.agent.train(self.model, hparam=self.hparam)) as agent:
            if outputs['discrete']:
                legal = old.get('legal', [None] * len(outputs['discrete']))
                assert len(legal) == len(outputs['discrete']), (len(legal), len(outputs['discrete']))
                for legal, logits, discrete in zip(legal, outputs['discrete'], torch.unbind(old['discrete'], -1)):
                    dist = agent.get_discrete_dist(logits[:-1], legal)
                    logp.append(dist.log_prob(discrete))
                    entropy.append(dist.entropy())
            if 'continuous' in outputs:
                dist = agent.get_continuous_dist([t[:-1] for t in outputs['continuous']])
                logp += torch.unbind(dist.log_prob(old['continuous']), -1)
                entropy += torch.unbind(dist.entropy(), -1)
        outputs['logp'] = logp = torch.stack(logp, -1)
        outputs['entropy'] = torch.stack(entropy, -1)
        outputs['ratio'] = torch.exp(logp - old['logp'])
        return outputs

    def estimate(self, old, new):
        with torch.no_grad():
            discount = torch.logical_not(old['done']).float() * self.discount
            advantage = torch.stack(ac.gae(old['reward'], new['critic'], new['critic_'], discount, self.gae))
            credit = new['critic'] + advantage
        return dict(advantage=advantage, credit=credit)

    def get_loss_policy(self, old, new):
        advantage = self.norm_advantage(new['advantage']).sum(-1)
        ratio = new['ratio'].mean(-1)
        return -torch.min(
            advantage * ratio,
            advantage * ratio.clamp(min=1 - self.hparam['clip'], max=1 + self.hparam['clip']),
        )

    def get_loss_critic(self, old, new):
        return self.loss_critic(new['critic'], new['credit'], reduction='none')

    def get_losses(self, old, new):
        policy = self.get_loss_policy(old, new)
        critic = self.get_loss_critic(old, new)
        assert len(policy.shape) == 1 or len(policy.shape) + 1 == len(critic.shape), (policy.shape, critic.shape)
        return dict(policy=policy.mean(), entropy=-new['entropy'].mean()), critic.view(-1, critic.shape[-1]).mean(0)

    def __call__(self):
        cost, old, results = next(self.prefetcher)
        self.cost += cost
        for _ in range(self.hparam['reuse']):
            self.optimizer.zero_grad()
            new = self.forward(old)
            new.update(self.estimate(old, new))
            losses, loss_critic = self.get_losses(old, new)
            loss = sum(loss * self.hparam[f"weight_loss_{key}"] for key, loss in losses.items()) + loss_critic.sum() * self.hparam['weight_loss_critic']
            loss.backward()
            self.optimizer.step()
            self.iteration += 1
            self.broadcaster()
        return dict(results=results, loss=loss, losses=losses, loss_critic=loss_critic, ratio=new['ratio'])


@wrap.record.stale(NAME)
class _Evaluator(Learner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        encoding = self.describe()['blob']
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
            lambda outcome: lamarckian.util.record.Scalar(self.cost, **{
                **{f'{NAME}/loss': outcome['loss'].item()},
                **{f'{NAME}/losses/{key}': loss.item() for key, loss in outcome['losses'].items()},
                **{f'{NAME}/losses/critic': outcome['loss_critic'].sum().item()},
                **{f'{NAME}/losses/critic_{key}': loss.item() for key, loss in zip(encoding['reward'], outcome['loss_critic'])},
            }),
        )
        self.recorder.register(
            lamarckian.util.counter.Time(**glom.glom(kwargs['config'], 'record.scalar')),
            lambda outcome: lamarckian.util.record.Scalar(self.cost, **record.get_ratio(self, outcome, NAME)),
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
        self.results += outcome['results']
        self.recorder(outcome)
        return outcome

    @lamarckian.util.duration.wrap(f"{NAME}/sync_blob")
    def sync_blob(self, *args, **kwargs):
        return super().sync_blob(*args, **kwargs)


class Evaluator(_Evaluator):
    pass


class DDP(lamarckian.rl.wrap.remote.ddp(_wrap.remote.training_switch(Learner))(RL)):
    def __init__(self, state={}, **kwargs):
        super().__init__(state, **kwargs)
        encoding = self.describe()['blob']
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
            lamarckian.rl.record.Rollout.counter(**kwargs),
            lambda *args, **kwargs: lamarckian.rl.record.Rollout.new(**kwargs)(self.cost, self.get_blob()),
        )
        self.recorder.register(
            lamarckian.util.counter.Time(**glom.glom(kwargs['config'], 'record.scalar')),
            lambda *args, **kwargs: lamarckian.util.record.Scalar(self.cost, **self.results()),
        )
        self.recorder.register(
            lamarckian.util.counter.Time(**glom.glom(kwargs['config'], 'record.scalar')),
            lambda outcome: lamarckian.util.record.Scalar(self.cost, **{
                **{f'{NAME}/loss': outcome['loss'].item()},
                **{f'{NAME}/losses/{key}': loss.item() for key, loss in outcome['losses'].items()},
                **{f'{NAME}/losses/critic': outcome['loss_critic'].sum().item()},
                **{f'{NAME}/losses/critic_{key}': loss.item() for key, loss in zip(encoding['reward'], outcome['loss_critic'])},
            }),
        )
        self.recorder.register(
            lamarckian.util.counter.Time(**glom.glom(kwargs['config'], 'record.scalar')),
            lambda outcome: lamarckian.util.record.Scalar(self.cost, **record.get_ratio(self, outcome, NAME)),
        )

    def close(self):
        self.saver()
        self.recorder.close()
        return super().close()

    def training(self):
        self.results = lamarckian.rl.record.Results(**self.kwargs)
        return super().training()

    def __call__(self):
        outcome = super().__call__()
        self.results += outcome['results']
        self.recorder(outcome)
        return outcome
