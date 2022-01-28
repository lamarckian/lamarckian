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
import pickle
import hashlib
import codecs
import asyncio

import numpy as np
import torch
import torch.nn.functional as F
import glom

import lamarckian
from lamarckian.mdp import rollout
from .. import pg

NAME = os.path.basename(os.path.dirname(__file__))


def cumulate(reward, discount, terminal=0):
    credit = [terminal]
    for r, gamma in zip(reversed(reward), reversed(discount)):
        credit.insert(0, r + gamma * credit[0])
    return credit[:-1]


def disassemble(credit, discount, terminal=0):
    reward = []
    for value, gamma in zip(reversed(credit), reversed(discount)):
        reward.insert(0, value - gamma * terminal)
        terminal = value
    return reward


def gae(reward, value, value_, discount, lmd):
    td = reward + discount * value_ - value
    return cumulate(td, discount * lmd)


class Actor(pg.Actor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        encoding = self.describe()['blob']
        self.hparam.setup('truncation', glom.glom(kwargs['config'], f'rl.{NAME}.truncation', default=np.iinfo(np.int).max), np.int)
        self.norm_advantage = eval('lambda advantage: ' + glom.glom(kwargs['config'], f'rl.{NAME}.norm_advantage', default='advantage'))
        gae = glom.glom(kwargs['config'], f'rl.{NAME}.gae', default=1)
        for name in encoding['reward']:
            self.hparam.setup(f"gae_{name}", glom.glom(kwargs['config'], f'rl.{NAME}.gae_{name}', default=gae), np.float)

    def describe_model(self):
        return glom.glom(self.kwargs['config'], 'model.backbone') + glom.glom(self.kwargs['config'], 'model.wrap', default=[]) + glom.glom(self.kwargs['config'], 'model.critic.wrap', default=[])

    def create_truncator(self):
        return lamarckian.rl.Truncator(self, self.hparam['truncation'])

    def training(self):
        training = super().training()
        self.truncator = self.create_truncator()
        return lamarckian.util.Closing(self.truncator, training)

    def rollout_mc(self):
        me, opponent = self.spawn()
        loop = asyncio.get_event_loop()
        battle = self.mdp.reset(me, *opponent, loop=loop)
        with contextlib.closing(battle), contextlib.closing(self.agent.train(self.model, hparam=self.hparam, generator=self.generator, **self.kwargs)) as agent, contextlib.closing(self.make_agents(opponent)) as agents:
            trajectory, exp = loop.run_until_complete(asyncio.gather(
                rollout.get_trajectory(battle.controllers[0], agent, cast=self.truncator.cast),
                *[rollout.get_cost(controller, agent) for controller, agent in zip(battle.controllers[1:], agents.values())],
                *battle.ticks,
            ))[0]
            result = battle.controllers[0].get_result()
            if opponent:
                result['digest_opponent_train'] = hashlib.md5(pickle.dumps(list(opponent.values()))).hexdigest()
            return trajectory, result

    def rollout(self):
        if self.truncator.step < np.iinfo(np.int).max:
            trajectory, exp = next(self.truncator)
            if self.truncator.done:
                result = self.truncator.battle.controllers[0].get_result()
                if self.truncator.opponent:
                    result['digest_opponent_train'] = hashlib.md5(pickle.dumps(list(self.truncator.opponent.values()))).hexdigest()
                results = [result]
            else:
                results = []
            return trajectory, exp, results
        else:
            trajectory, result = self.rollout_mc()
            return trajectory, None, [result]

    def to_tensor(self, trajectory, exp):
        tensors = {}
        try:
            tensors['legal'] = [torch.cat([exp['legal'][i] for exp in trajectory]) for i in range(len(trajectory[0]['legal']))]
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
        tensors['reward'] = self.norm_reward(reward)
        tensors['done'] = torch.from_numpy(np.array([exp['done'] for exp in trajectory], np.bool)).unsqueeze(-1).to(self.device)
        return tensors


class Learner(Actor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparam.setup('lr', glom.glom(kwargs['config'], 'train.lr'), np.float)
        for key in 'policy critic entropy'.split():
            self.hparam.setup(f'weight_loss_{key}', glom.glom(kwargs['config'], f"rl.{NAME}.weight_loss.{key}"), np.float)
        self.loss_critic = getattr(F, glom.glom(kwargs['config'], f"rl.{NAME}.loss_critic", default='mse_loss'))

    def training(self):
        var = {}
        with codecs.open(os.path.join(os.path.dirname(os.path.abspath(lamarckian.__file__)), 'import.py'), 'r', 'utf-8') as f:
            exec(f.read(), var)
        self.optimizer = eval('lambda params, lr: ' + glom.glom(self.kwargs['config'], 'train.optimizer'), var)(filter(lambda p: p.requires_grad, self.model.parameters()), self.hparam['lr'])
        encoding = self.describe()['blob']
        self.gae = torch.FloatTensor([self.hparam[f"gae_{name}"] for name in encoding['reward']]).to(self.device)
        return super().training()

    def get_entropy(self, trajectory):
        entropy = []
        for i in range(len(trajectory[0]['discrete'])):
            entropy.append(torch.cat([exp['discrete_dist'][i].entropy() for exp in trajectory]))
        try:
            entropy += torch.unbind(torch.cat([exp['continuous_dist'].entropy() for exp in trajectory]), -1)
        except KeyError:
            pass
        return torch.stack(entropy, -1)

    def to_tensor(self, trajectory, exp):
        tensors = super().to_tensor(trajectory, exp)
        tensors['critic'] = critic = torch.cat([exp['critic'] for exp in trajectory])
        with torch.no_grad():
            discount = torch.logical_not(tensors['done']).float() * self.discount
            terminal = torch.zeros(1, discount.shape[-1]).to(self.device) if exp is None else self.model(*exp['inputs'])['critic']
            if (self.gae < 1).any():
                tensors['advantage'] = advantage = torch.stack(gae(tensors['reward'], critic, torch.cat([critic[1:], terminal]), discount, self.gae))
                tensors['credit'] = critic + advantage
            else:
                tensors['credit'] = credit = torch.stack(cumulate(tensors['reward'], discount, terminal.squeeze(0)))
                tensors['advantage'] = credit - critic
        tensors['entropy'] = -self.get_entropy(trajectory)
        return tensors

    def get_losses(self, tensors):
        advantage = self.norm_advantage(tensors['advantage']).sum(-1)
        policy = -tensors['logp'].mean(-1) * advantage
        critic = self.loss_critic(tensors['critic'], tensors['credit'], reduction='none')
        return dict(policy=policy.mean(), entropy=tensors['entropy'].mean()), critic.mean(0)

    def backward(self):
        trajectory, terminal, results = self.rollout()
        self.cost += sum(exp.get('cost', 1) for exp in trajectory)
        tensors = self.to_tensor(trajectory, terminal)
        losses, loss_critic = self.get_losses(tensors)
        loss = sum(loss * self.hparam[f"weight_loss_{key}"] for key, loss in losses.items()) + loss_critic.sum() * self.hparam['weight_loss_critic']
        loss.backward()
        return dict(results=results, loss=loss, losses=losses, loss_critic=loss_critic)

    def __call__(self):
        self.optimizer.zero_grad()
        outcome = self.backward()
        self.optimizer.step()
        return outcome


class Evaluator(Learner):
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
