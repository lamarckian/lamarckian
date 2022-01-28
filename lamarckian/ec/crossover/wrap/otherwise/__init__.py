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
import copy

import numpy as np


def all(crossover):
    class Crossover(crossover):
        def __call__(self, *args, **kwargs):
            if self.random.random() < self.kwargs['prob']:
                return super().__call__(*args, **kwargs)
            else:
                return tuple(map(copy.deepcopy, args))
    Crossover.__call__.__signature__ = inspect.signature(crossover.__call__)
    return Crossover


def best(crossover):
    class Crossover(crossover):
        def __call__(self, *args, **kwargs):
            if self.random.random() < self.kwargs['prob']:
                return super().__call__(*args, **kwargs)
            else:
                assert len(args) > 1, len(args)
                index = np.argmax([parent['result']['fitness'] for parent in kwargs['ancestor']])
                return copy.deepcopy(args[index]),
    Crossover.__call__.__signature__ = inspect.signature(crossover.__call__)
    return Crossover
