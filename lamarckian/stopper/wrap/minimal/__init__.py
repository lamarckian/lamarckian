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
import time

import humanfriendly
import glom


def cost(begin='stopper.minimal.cost'):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    def decorate(stopper):
        class Stopper(stopper):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, PATH_FUNC)
                setattr(self, PATH_FUNC, self.evaluator.cost + (humanfriendly.parse_size(str(glom.glom(kwargs['config'], begin))) if isinstance(begin, str) else begin))

            def __call__(self, *args, **kwargs):
                if self.evaluator.cost > getattr(self, PATH_FUNC):
                    return super().__call__(*args, **kwargs)
                return False
        return Stopper
    return decorate


def iteration(begin='stopper.minimal.iteration'):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    def decorate(stopper):
        class Stopper(stopper):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, PATH_FUNC)
                setattr(self, PATH_FUNC, self.evaluator.iteration + (humanfriendly.parse_size(str(glom.glom(kwargs['config'], begin))) if isinstance(begin, str) else begin))

            def __call__(self, *args, **kwargs):
                if self.evaluator.iteration > getattr(self, PATH_FUNC):
                    return super().__call__(*args, **kwargs)
                return False
        return Stopper
    return decorate


def duration(begin='stopper.minimal.duration'):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    def decorate(stopper):
        class Stopper(stopper):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, PATH_FUNC)
                try:
                    interval = glom.glom(kwargs['config'], begin)
                except ValueError:
                    interval = begin
                attr = types.SimpleNamespace(start=time.time(), min=humanfriendly.parse_timespan(interval))
                assert attr.min > 0, attr.min
                setattr(self, PATH_FUNC, attr)

            def __call__(self, *args, **kwargs):
                attr = getattr(self, PATH_FUNC)
                if time.time() - attr.start > attr.min:
                    return super().__call__(*args, **kwargs)
                return False
        return Stopper
    return decorate
