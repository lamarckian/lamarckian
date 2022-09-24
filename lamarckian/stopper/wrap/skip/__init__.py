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
import functools

import humanfriendly
import glom

import lamarckian


def iteration(interval='stopper.skip.iteration'):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    def decorate(stopper):
        class Stopper(stopper):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, PATH_FUNC)
                setattr(self, PATH_FUNC, lamarckian.util.counter.Number(humanfriendly.parse_size(str(glom.glom(kwargs['config'], interval))) if isinstance(interval, str) else interval))

            def __call__(self, *args, **kwargs):
                if getattr(self, PATH_FUNC)():
                    return super().__call__(*args, **kwargs)
                return False
        return Stopper
    return decorate


def cost(interval='stopper.skip.cost'):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    def decorate(stopper):
        class Stopper(stopper):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, PATH_FUNC)
                setattr(self, PATH_FUNC, lamarckian.util.counter.Number(humanfriendly.parse_size(str(glom.glom(kwargs['config'], interval))) if isinstance(interval, str) else interval))

            def __call__(self, outcome, *args, **kwargs):
                if getattr(self, PATH_FUNC)(outcome['cost']):
                    return super().__call__(outcome, *args, **kwargs)
                return False
        return Stopper
    return decorate


def duration(interval='stopper.skip.duration'):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    def decorate(stopper):
        class Stopper(stopper):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, PATH_FUNC)
                setattr(self, PATH_FUNC, lamarckian.util.counter.Time(humanfriendly.parse_timespan(str(glom.glom(kwargs['config'], interval))) if isinstance(interval, str) else interval))

            def __call__(self, *args, **kwargs):
                if getattr(self, PATH_FUNC)():
                    return super().__call__(*args, **kwargs)
                return False
        return Stopper
    return decorate
