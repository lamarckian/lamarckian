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

import inspect
import types

import numpy as np
import torch
import glom

import lamarckian
from . import rpc, record


def lstm(rl):
    learner = rl.Learner if hasattr(rl, 'Learner') else rl

    class Learner(learner):
        def forward(self, old, *args, **kwargs):
            with torch.no_grad():
                inputs = tuple(input.view(np.multiply.reduce(input.shape[:2]), *input.shape[2:]) for input in (input[:-1] for input in old['inputs']))
                outputs = self.model(*inputs)
                hidden = outputs['hidden']
                for input, output in zip(old['inputs'][-len(hidden):], hidden):
                    input[1:] = output.view(-1, *input.shape[1:])
            return super().forward(old, *args, **kwargs)

    if hasattr(rl, 'Learner'):
        rl.Learner = Learner
        return rl
    else:
        return Learner


def learn(rl):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    learner = rl.Learner if hasattr(rl, 'Learner') else rl

    def wrap_actor(actor):
        to_tensor = actor.to_tensor

        class Actor(actor):
            def to_tensor(self, trajectory, *args, **kwargs):
                tensors = to_tensor(self, trajectory, *args, **kwargs)
                tensors[NAME_FUNC] = torch.from_numpy(np.array([exp.get(NAME_FUNC, True) for exp in trajectory], np.bool)).unsqueeze(1).to(self.device)
                return tensors
        return Actor

    def get_record(self):
        stat = getattr(self, PATH_FUNC)
        return {NAME_FUNC: stat.count / stat.total}

    class Learner(learner):
        Actor = wrap_actor(learner.Actor)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            setattr(self, PATH_FUNC, types.SimpleNamespace(count=0, total=0))
            self.recorder.register(
                lamarckian.util.counter.Time(**glom.glom(kwargs['config'], 'record.scalar')),
                lambda outcome: lamarckian.util.record.Scalar(self.cost, **get_record(self)),
            )

        def __next__(self):
            cost, tensors, results, iterations = super().__next__()
            stat = getattr(self, PATH_FUNC)
            mask = tensors[NAME_FUNC]
            stat.count += mask.sum().item()
            stat.total += np.multiply.reduce(mask.shape)
            return cost, tensors, results, iterations

        def get_loss_policy(self, old, new):
            loss = super().get_loss_policy(old, new)
            mask = ~old[NAME_FUNC]
            loss[mask] = loss[mask].detach()
            try:
                new['entropy'][mask] = new['entropy'][mask].detach()
            except KeyError:
                pass
            return loss

    if hasattr(rl, 'Learner'):
        rl.Learner = Learner
        return rl
    else:
        return Learner


def fresh(rl):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    class RL(rl):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            setattr(self, PATH_FUNC, glom.glom(kwargs['config'], 'rl.ppo.fresh'))

        def receive(self):
            while True:
                cost, tensors, results, iteration = self.rpc_gather()
                if self.iteration - iteration <= getattr(self, PATH_FUNC):
                    tensors = lamarckian.rl.to_device(self.device, **tensors)
                    return cost, tensors, results, iteration
    return RL
