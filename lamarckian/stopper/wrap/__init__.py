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

import glom

import lamarckian
from . import fitness, objective, maximal, minimal, skip, outcome, evaluate


def patience(value='stopper.patience'):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    def decorate(stopper):
        class Stopper(stopper):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, PATH_FUNC)
                setattr(self, PATH_FUNC, lamarckian.util.counter.Number(glom.glom(kwargs['config'], value) if isinstance(value, str) else value))

            def __call__(self, *args, **kwargs):
                attr = getattr(self, PATH_FUNC)
                if super().__call__(*args, **kwargs):
                    return attr()
                else:
                    attr.reset()
                    return False
        return Stopper
    return decorate
