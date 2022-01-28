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

import numpy as np
import torch
import torch.nn.functional as F
import glom

import lamarckian
from .. import Remote, ac

NAME = os.path.basename(os.path.dirname(__file__))


class Actor(lamarckian.util.rpc.wrap.any(lamarckian.util.rpc.wrap.all(ac.Learner))):
    def __init__(self, *args, **kwargs):
        torch.set_num_threads(1)
        super().__init__(*args, **kwargs)

    def gradient(self, blob):
        self.set_blob(blob)
        cost = self.cost
        self.optimizer.zero_grad()
        outcome = self.backward()
        outcome['gradient'] = [param.grad.cpu().numpy() for param in self.model.parameters()]
        return self.cost - cost, outcome


class Learner(Remote):
    Actor = Actor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_critic = getattr(F, glom.glom(kwargs['config'], f'rl.ac.loss_critic', default='mse_loss'))
        self.rpc_gradient = lamarckian.util.rpc.wrap.any_count(lamarckian.util.rpc.Any)(self.actors, **kwargs)

    def close(self):
        self.rpc_gradient.close()
        return super().close()

    def training(self):
        training = super().training()
        blob = self.get_blob()
        for _ in range(len(self.actors)):
            self.rpc_gradient.send('gradient', blob)

        def close():
            for _ in range(len(self.rpc_gradient)):
                self.rpc_gradient.receive()
            return training.close()
        return types.SimpleNamespace(close=close)

    def __call__(self):
        cost, outcome = self.rpc_gradient.receive()
        self.rpc_gradient.send('gradient', self.get_blob())
        self.cost += cost
        self.optimizer.zero_grad()
        for param, grad in zip(self.model.parameters(), outcome['gradient']):
            param.grad = torch.from_numpy(np.array(grad))
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
