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


def resample(explore):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    class Explore(explore):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            setattr(self, PATH_FUNC, types.SimpleNamespace(
                prob=glom.glom(kwargs['config'], f'ec.mutation.real.{NAME_FUNC}'),
                rs=np.random.RandomState(glom.glom(kwargs['config'], 'seed', default=None)),
            ))

        def __call__(self, parent, **kwargs):
            attr = getattr(self, PATH_FUNC)
            if attr.rs.rand() < attr.prob:
                return self.lower + attr.rs.random(self.lower.shape) * (self.upper - self.lower)
            else:
                return super().__call__(parent, **kwargs)
    return Explore
