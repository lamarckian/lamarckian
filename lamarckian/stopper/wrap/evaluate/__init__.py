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
import collections

import glom


def train(stopper):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    class Stopper(stopper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            setattr(self, PATH_FUNC, collections.deque(maxlen=glom.glom(kwargs['config'], 'sample.train')))

        def __call__(self, outcome, *args, **kwargs):
            attr = getattr(self, PATH_FUNC)
            attr += outcome['results']
            return super().__call__(outcome, *args, **kwargs) and attr

        def evaluate(self):
            attr = getattr(self, PATH_FUNC)
            assert attr
            return list(attr)
    return Stopper
