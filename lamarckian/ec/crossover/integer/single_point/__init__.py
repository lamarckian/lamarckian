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

import random

import numpy as np
import glom

from .. import Crossover as _Crossover


class Crossover(_Crossover):
    def __init__(self, encoding, **kwargs):
        super().__init__(encoding, **kwargs)
        self.random = random.Random(glom.glom(kwargs['config'], 'seed', default=None))
        range = self.upper - self.lower + 1
        self.bits = np.ceil(np.log2(range)).astype(range.dtype)
        self.mask = np.power(2, self.bits) - 1
        assert self.mask.dtype == range.dtype
        assert np.all(np.greater_equal(self.mask, self.upper - self.lower))

    def __call__(self, parent1, parent2, **kwargs):
        assert len(parent1.shape) == 1
        assert parent1.shape == parent2.shape
        assert parent1.dtype == self.mask.dtype, (parent1.dtype, self.mask.dtype)
        assert parent2.dtype == self.mask.dtype, (parent2.dtype, self.mask.dtype)
        return np.vectorize(self.crossover)(parent1, parent2, self.lower, self.upper, self.bits, self.mask)

    def crossover(self, parent1, parent2, lower, upper, bits, mask):
        position = self.random.randint(0, bits - 1)
        mask_upper = mask << position
        mask_lower = ~mask_upper
        assert lower <= parent1 <= upper, (lower, parent1, upper)
        assert lower <= parent2 <= upper, (lower, parent2, upper)
        _parent1 = parent1 - lower
        _parent2 = parent2 - lower
        _child1 = (_parent1 & mask_upper) | (_parent2 & mask_lower)
        _child2 = (_parent1 & mask_lower) | (_parent2 & mask_upper)
        child1 = np.clip(lower + _child1, lower, upper)
        child2 = np.clip(lower + _child2, lower, upper)
        if self.random.random() < 0.5:
            return child1, child2
        else:
            return child2, child1
