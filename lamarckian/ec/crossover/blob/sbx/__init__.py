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

import os

import numpy as np
import glom

from .. import Crossover as _Crossover
NAME = os.path.basename(os.path.dirname(__file__))


class Crossover(_Crossover):
    def __init__(self, encoding, **kwargs):
        super().__init__(encoding, **kwargs)
        self.rs = np.random.RandomState(glom.glom(kwargs['config'], 'seed', default=None))
        self.distribution_index = kwargs[NAME]['distribution_index']

    def __call__(self, parent1, parent2, **kwargs):
        return list(zip(*[self.crossover(value1, value2) for value1, value2 in zip(parent1, parent2)]))

    def crossover(self, parent1, parent2):
        assert parent1.shape == parent2.shape
        assert self.distribution_index >= 0, self.distribution_index
        u = self.rs.rand(*parent1.shape)
        spread_factor = np.vectorize(lambda u: (2 * u) ** (1 / (self.distribution_index + 1)) if u < 0.5 else (1 / 2 / (1 - u)) ** (1 / (self.distribution_index + 1)))(u)
        middle = (parent1 + parent2) / 2
        dist = np.abs(parent1 - parent2)
        dist2 = dist / 2
        sign = np.sign(self.rs.rand(*parent1.shape) - 0.5)
        diff2 = spread_factor * dist2 * sign
        child1 = middle - diff2
        child2 = middle + diff2
        if self.rs.random() < 0.5:
            return child1, child2
        else:
            return child2, child1
