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

import numpy as np
import torch
import glom

import lamarckian
from .. import RL, ppo, disc, wrap as _wrap

NAME = os.path.basename(os.path.dirname(__file__))


def v_trace(reward, critic, critic_, discount, ratio, **kwargs):
    td = reward + discount * critic_ - critic
    advantage = [torch.zeros_like(critic_[-1])]
    for delta, gamma, trace in zip(reversed(ratio.clamp(max=kwargs['rho']) * td), reversed(discount), reversed(ratio.clamp(max=kwargs['trace']))):
        advantage.insert(0, delta + gamma * trace * advantage[0])
    return advantage[:-1]


class Learner(disc.Learner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparam.setup('clip_rho', glom.glom(self.kwargs['config'], f'rl.{NAME}.clip', default=np.finfo(np.float32).max), np.float)

    def estimate(self, old, new):
        critic, critic_ = new['critic'], new['critic_']
        ratio = new['ratio'].mean(-1).unsqueeze(-1)
        with torch.no_grad():
            discount = torch.logical_not(old['done']).float() * self.discount
            new['credit'] = credit = critic + torch.stack(v_trace(old['reward'], critic, critic_, discount, ratio, rho=self.hparam['clip_rho'], trace=self.hparam['clip_trace']))
            credit_ = torch.cat([credit[1:], critic_[-1:]])
            advantage = old['reward'] + discount * credit_ - critic
        return dict(advantage=advantage, credit=credit)

    def get_loss_policy(self, old, new):
        advantage = self.norm_advantage(new['advantage']).sum(-1)
        ratio = new['ratio'].clamp(max=self.hparam['clip']).mean(-1).detach()
        return -new['logp'].mean(-1) * advantage * ratio


@ppo.wrap.record.stale(NAME)
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
            lambda outcome: lamarckian.util.record.Scalar(self.cost, **ppo.record.get_ratio(self, outcome, NAME)),
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
