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
import copy
import types

import numpy as np
import torch

import lamarckian


def best(tag='stopper/fitness//checkpoint'):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    def decorate(stopper):
        class Stopper(stopper):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, PATH_FUNC)
                setattr(self, PATH_FUNC, types.SimpleNamespace(
                    best=np.finfo(np.float).min,
                ))
                self.evaluator.recorder.put(lamarckian.util.record.Scalar(self.evaluator.cost, **{tag: np.nan}))

            def __call__(self, outcome, *args, **kwargs):
                fitness = kwargs['fitness']
                attr = getattr(self, PATH_FUNC)
                if fitness > attr.best and not torch.isnan(outcome['loss']):
                    attr.best = fitness
                    attr.decision = copy.deepcopy(self.evaluator.get())
                    self.evaluator.recorder.put(lamarckian.util.record.Scalar(self.evaluator.cost, **{tag: fitness}))
                return super().__call__(outcome, *args, **kwargs)

            def get(self):
                attr = getattr(self, PATH_FUNC)
                try:
                    return attr.decision
                except AttributeError:
                    return self.evaluator.get()
        return Stopper
    return decorate
