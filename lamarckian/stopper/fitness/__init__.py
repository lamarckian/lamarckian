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

import numpy as np
import glom

from .. import Stopper as _Stopper


class Improve(_Stopper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best = np.finfo(np.float).min

    def __call__(self, *args, **kwargs):
        fitness = kwargs['fitness']
        if fitness > self.best:
            self.best = fitness
            return False
        else:
            return True


def greater(value='stopper.fitness'):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    class Stopper(_Stopper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            setattr(self, PATH_FUNC, glom.glom(kwargs['config'], value) if isinstance(value, str) else value)

        def __call__(self, *args, **kwargs):
            fitness = kwargs['fitness']
            if fitness > getattr(self, PATH_FUNC):
                return True
            return False
    return Stopper
