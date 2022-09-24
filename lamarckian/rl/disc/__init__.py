"""
Copyright (C) 2020, 申瑞珉 (Ruimin Shen)

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
import humanfriendly

import lamarckian
from .. import RL, ppo, wrap as _wrap

NAME = os.path.basename(os.path.dirname(__file__))


def gae_v(reward, critic, critic_, discount, lmd, ratio):
    td = reward + discount * critic_ - critic
    last = torch.zeros_like(critic_[-1])
    advantage = []
    for delta, gamma, c in zip(reversed(td), reversed(discount), reversed(ratio)):
        advantage.insert(0, delta + gamma * lmd * last)
        last = c * advantage[0]
    return advantage


class Learner(ppo.Learner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparam.setup('clip_trace', glom.glom(self.kwargs['config'], f'rl.{NAME}.clip', default=1), np.float)

    def estimate(self, old, new):
        with torch.no_grad():
            ratio_ = new['ratio'].clamp(max=self.hparam['clip_trace']).mean(-1).unsqueeze(-1)
            discount = torch.logical_not(old['done']).to(old['reward'].dtype) * self.discount
            advantage = torch.stack(gae_v(old['reward'], new['critic'], new['critic_'], discount, self.gae, ratio_))
            credit = new['critic'] + ratio_ * advantage
        return dict(advantage=advantage, credit=credit)


@ppo.wrap.record.stale(NAME)
class _Evaluator(Learner):
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
            lamarckian.rl.record.Rollout.counter(**kwargs),
            lambda *args, **kwargs: lamarckian.rl.record.Rollout.new(**kwargs)(self.cost, self.get_blob()),
        )
        self.recorder.register(
            lamarckian.util.counter.Time(**glom.glom(kwargs['config'], 'record.scalar')),
            lambda outcome: lamarckian.util.record.Scalar(self.cost, **{
                **{f'{NAME}/loss': outcome['loss'].item()},
                **{f'{NAME}/losses/{key}': loss.item() for key, loss in outcome['losses'].items()},
                **{f'{NAME}/losses/critic': outcome['loss_critic'].sum().item()},
                **{f'{NAME}/losses/critic_{key}': loss.item() for key, loss in zip(self.encoding['blob']['reward'], outcome['loss_critic'])},
                **self.results(),
                **{f'{NAME}/{key}': value for key, value in ppo.record.get_ratio(self, outcome).items()},
                **{f'{NAME}/broadcast/{key}': value for key, value in getattr(self.broadcaster, 'profile', {}).items()},
            }),
        )
        self.recorder.register(
            lamarckian.util.counter.Time(**glom.glom(kwargs['config'], 'record.model')),
            lambda outcome: lamarckian.rl.record.Model(f"{NAME}/model", self.cost, {key: value.cpu().numpy() for key, value in self.model.state_dict().items()}, glom.glom(kwargs['config'], 'record.model.sort', default='bytes')),
        )
        self.recorder.put(lamarckian.rl.record.Graph(f"{NAME}/graph", self.cost))
        self.recorder.put(lamarckian.util.record.Text(self.cost, **{f"{NAME}/model_size": humanfriendly.format_size(sum(value.cpu().numpy().nbytes for value in self.model.state_dict().values()))}))
        self.recorder.put(lamarckian.util.record.Text(self.cost, topology=repr(self.rpc_all)))

    def close(self):
        self.saver.close()
        super().close()
        self.recorder.close()

    def training(self):
        self.results = lamarckian.rl.record.Results(**self.kwargs)
        return super().training()

    @lamarckian.util.duration.wrap('step')
    def __call__(self):
        outcome = super().__call__()
        self.results += outcome['results']
        self.recorder(outcome)
        return outcome


class Evaluator(_Evaluator):  # ray's bug
    pass


@ppo.wrap.record.ddp.topology
@ppo.wrap.record.ddp.stale(NAME)
@ppo.wrap.record.ddp.broadcaster(NAME)
class DDP(lamarckian.rl.wrap.remote.ddp(_wrap.remote.training_switch(Learner))(RL)):
    def __init__(self, *args, **kwargs):
        broadcaster0 = glom.glom(kwargs['config'], 'ddp.broadcaster0', default=True)
        super().__init__(*args, **kwargs, broadcaster=not broadcaster0)
        if broadcaster0:
            self.ray.get(self.learners[0].create_broadcaster.remote(self.actors))
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
            lambda outcome: lamarckian.util.record.Scalar(self.cost, **{
                **{f'{NAME}/loss': outcome['loss'].item()},
                **{f'{NAME}/losses/{key}': loss.item() for key, loss in outcome['losses'].items()},
                **{f'{NAME}/losses/critic': outcome['loss_critic'].sum().item()},
                **{f'{NAME}/losses/critic_{key}': loss.item() for key, loss in zip(self.encoding['blob']['reward'], outcome['loss_critic'])},
                **self.results(),
                **{f'{NAME}/{key}': value for key, value in ppo.record.get_ratio(self, outcome).items()},
            }),
        )
        encoding = self.encoding['blob']
        model = encoding['models'][0]
        module = lamarckian.evaluator.parse(*model['cls'], **kwargs)
        keys = list(module(**model, **kwargs, reward=encoding['reward']).state_dict().keys())
        blob = self.get_blob()
        self.recorder.put(lamarckian.util.record.Text(self.cost, **{f"{NAME}/model_size": humanfriendly.format_size(sum(value.nbytes for value in blob))}))
        self.recorder.register(
            lamarckian.util.counter.Time(**glom.glom(kwargs['config'], 'record.model')),
            lambda outcome: lamarckian.rl.record.Model(f"{NAME}/model", self.cost, {key: value for key, value in zip(keys, self.get_blob())}, glom.glom(kwargs['config'], 'record.model.sort', default='bytes')),
        )

    def close(self):
        self.saver.close()
        super().close()
        self.recorder.close()

    def training(self):
        self.results = lamarckian.rl.record.Results(**self.kwargs)
        return super().training()

    @lamarckian.util.duration.wrap('step')
    def __call__(self):
        outcome = super().__call__()
        self.results += outcome['results']
        self.recorder(outcome)
        return outcome
