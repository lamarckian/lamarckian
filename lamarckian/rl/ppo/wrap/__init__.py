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

import inspect
import types

import numpy as np
import torch
import glom

import lamarckian
from . import record


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


def learnable(keys='advantage ratio entropy'.split()):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    def wrap_actor(actor):
        to_tensor = actor.to_tensor

        class Actor(actor):
            def to_tensor(self, trajectory, *args, **kwargs):
                tensors = to_tensor(self, trajectory, *args, **kwargs)
                tensors['learn'] = torch.from_numpy(np.array([exp.get('learn', True) for exp in trajectory], np.bool)).unsqueeze(-1).to(self.device)
                return tensors
        return Actor

    def decorate(rl):
        def get_record(self):
            stat = getattr(self.prefetcher, PATH_FUNC)
            return {NAME_FUNC: stat.count / stat.total}

        class RL(rl):
            Actor = wrap_actor(rl.Actor)

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, PATH_FUNC)
                self.recorder.register(
                    lamarckian.util.counter.Time(**glom.glom(kwargs['config'], 'record.scalar')),
                    lambda outcome: lamarckian.util.record.Scalar(self.cost, **get_record(self)),
                )

            def training(self):
                training = super().training()
                next = self.prefetcher.__next__
                class _(type(self.prefetcher)):
                    def __next__(self, *arg, **kwarg):
                        stat = types.SimpleNamespace(count=0, total=0)
                        while True:
                            cost, old, results = next()
                            mask = old['learn']
                            stat.count += mask.sum().item()
                            stat.total += np.multiply.reduce(mask.shape)
                            if old['learn'].any().item():
                                break
                        setattr(self, PATH_FUNC, stat)
                        return cost, old, results
                self.prefetcher.__class__ = _
                return training

            def get_losses(self, old, new):
                mask = old['learn']
                for key in keys:
                    try:
                        new[key] = new[key][mask]
                    except KeyError:
                        pass
                return super().get_losses(old, new)
        return RL
    return decorate
