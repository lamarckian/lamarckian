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
import humanfriendly

import lamarckian
from .. import Stopper as _Stopper


def relative(end='stopper.iteration'):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    class Stopper(_Stopper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            setattr(self, PATH_FUNC, lamarckian.util.counter.Number(humanfriendly.parse_size(str(glom.glom(kwargs['config'], end))) if isinstance(end, str) else end))

        def __call__(self, *args, **kwargs):
            return getattr(self, PATH_FUNC)()

        def __repr__(self):
            return f'iteration={repr(getattr(self, PATH_FUNC))}'
    return Stopper


def absolute(end='stopper.iteration'):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    class Stopper(_Stopper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            setattr(self, PATH_FUNC, humanfriendly.parse_size(str(glom.glom(kwargs['config'], end))) if isinstance(end, str) else end)

        def __call__(self, outcome, *args, **kwargs):
            return self.evaluator.iteration >= getattr(self, PATH_FUNC)
    return Stopper


def repeat_last(end='stopper.repeat_last'):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    class Stopper(_Stopper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            if isinstance(end, str):
                ends = glom.glom(kwargs['config'], end)
            else:
                assert isinstance(end, collections.Iterable)
                ends = end
            setattr(self, PATH_FUNC, lamarckian.util.counter.wrap.repeat_last(*ends)(lamarckian.util.counter.Number)())

        def __call__(self, *args, **kwargs):
            return getattr(self, PATH_FUNC)()
    return Stopper
