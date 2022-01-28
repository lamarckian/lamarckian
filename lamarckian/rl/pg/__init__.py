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
import contextlib
import codecs
import pickle
import hashlib
import asyncio

import numpy as np
import torch
import glom
import deepmerge

import lamarckian
from lamarckian.mdp import rollout
from .. import RL1

NAME = os.path.basename(os.path.dirname(__file__))

from . import agent


class Actor(RL1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        encoding = self.describe()['blob']
        discount = glom.glom(kwargs['config'], 'rl.discount')
        for name in encoding['reward']:
            self.hparam.setup(f"discount_{name}", glom.glom(kwargs['config'], f"rl.discount_{name}", default=discount), np.float)
        self.hparam.setup('prob_min', glom.glom(kwargs['config'], f'rl.{NAME}.prob_min', default=0), np.float)
        self.norm_reward = eval('lambda reward: ' + glom.glom(kwargs['config'], f'rl.{NAME}.norm_reward', default='reward'))

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

    def training(self):
        encoding = self.describe()['blob']
        self.discount = torch.FloatTensor([self.hparam[f"discount_{name}"] for name in encoding['reward']]).to(self.device)
        assert torch.logical_and(0 < self.discount, self.discount <= 1).all().item(), self.discount.numpy()
        return super().training()

    def rollout(self):
        me, opponent = self.spawn()
        loop = asyncio.get_event_loop()
        battle = self.mdp.reset(me, *opponent, loop=loop)
        with contextlib.closing(battle), contextlib.closing(self.agent.train(self.model, hparam=self.hparam, generator=self.generator, **self.kwargs)) as agent, contextlib.closing(self.make_agents(opponent)) as agents:
            trajectory, exp = loop.run_until_complete(asyncio.gather(
                rollout.get_trajectory(battle.controllers[0], agent),
                *[rollout.get_cost(controller, agent) for controller, agent in zip(battle.controllers[1:], agents.values())],
                *battle.ticks,
            ))[0]
            result = battle.controllers[0].get_result()
            if opponent:
                result['digest_opponent_train'] = hashlib.md5(pickle.dumps(list(opponent.values()))).hexdigest()
            return trajectory, [result]

    def to_tensor(self, trajectory):
        tensors = {}
        try:
            tensors['legal'] = torch.cat([exp['legal'] for exp in trajectory])
        except KeyError:
            pass
        # action
        try:
            tensors['discrete'] = torch.stack([torch.cat([exp['discrete'][i] for exp in trajectory]) for i in range(len(trajectory[0]['discrete']))], -1)
        except RuntimeError:
            pass
        try:
            tensors['continuous'] = torch.cat([exp['continuous'] for exp in trajectory])
        except KeyError:
            pass
        # logp
        logp = [torch.cat([exp['discrete_dist'][i].log_prob(exp['discrete'][i]) for exp in trajectory]) for i in range(len(trajectory[0]['discrete']))]
        try:
            logp += torch.unbind(torch.cat([exp['continuous_dist'].log_prob(exp['continuous']) for exp in trajectory]), -1)
        except KeyError:
            pass
        tensors['logp'] = torch.stack(logp, -1)
        # reward
        reward = torch.FloatTensor(np.array([exp['reward'] for exp in trajectory])).to(self.device)
        tensors['reward'] = reward = self.norm_reward(reward)
        tensors['credit'] = torch.stack(lamarckian.rl.cumulate(reward, self.discount))
        return tensors

    def __next__(self):
        trajectory, results = self.rollout()
        self.cost += sum(exp.get('cost', 1) for exp in trajectory)
        return self.to_tensor(trajectory), results

    def __getstate__(self):
        state = super().__getstate__()
        state['mdp'] = self.mdp.__getstate__()
        return state


class Learner(Actor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparam.setup('lr', glom.glom(kwargs['config'], 'train.lr'), np.float)
        glom.glom(kwargs['config'], f'rl.{NAME}.norm_reward', default='reward')

    def training(self):
        var = {}
        with codecs.open(os.path.join(os.path.dirname(os.path.abspath(lamarckian.__file__)), 'import.py'), 'r', 'utf-8') as f:
            exec(f.read(), var)
        self.optimizer = eval('lambda params, lr: ' + glom.glom(self.kwargs['config'], 'train.optimizer'), var)(filter(lambda p: p.requires_grad, self.model.parameters()), self.hparam['lr'])
        return super().training()

    def backward(self):
        tensors, results = next(self)
        logp = tensors['logp'].mean(-1)
        credit = tensors['credit'].sum(-1)
        loss = (-logp * credit).mean()
        loss.backward()
        return dict(results=results, loss=loss)

    def __call__(self):
        self.optimizer.zero_grad()
        outcome = self.backward()
        self.optimizer.step()
        return outcome


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
