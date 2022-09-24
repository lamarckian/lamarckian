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
import glom

import lamarckian
from . import checkpoint, debug


def smooth(sample='sample.train', tag='stopper/smooth'):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    def decorate(stopper):
        class Stopper(stopper):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, PATH_FUNC)
                setattr(self, PATH_FUNC, types.SimpleNamespace(
                    sample=glom.glom(kwargs['config'], sample) if isinstance(sample, str) else sample,
                    recent=[],
                ))
                self.evaluator.recorder.put(lamarckian.util.record.Scalar(self.evaluator.cost, **{tag: np.nan}))

            def __call__(self, outcome, *args, **kwargs):
                attr = getattr(self, PATH_FUNC)
                attr.recent += [result['fitness'] for result in outcome['results']]
                if len(attr.recent) >= attr.sample:
                    fitness = np.mean(attr.recent)
                    attr.recent.clear()
                    self.evaluator.recorder.put(lamarckian.util.record.Scalar(self.evaluator.cost, **{f'{tag}/fitness': fitness}))
                    return super().__call__(outcome, *args, **kwargs, fitness=fitness)
                return False
        return Stopper
    return decorate
