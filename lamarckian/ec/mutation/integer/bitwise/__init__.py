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

import functools
import random

import numpy as np
import glom

from .. import Mutation as _Mutation


class Mutation(_Mutation):
    def __init__(self, encoding, **kwargs):
        super().__init__(encoding, **kwargs)
        range = self.upper - self.lower + 1
        self.bits = np.ceil(np.log2(range)).astype(range.dtype)
        self.random = random.Random(glom.glom(kwargs['config'], 'seed', default=None))

    def __call__(self, parent, **kwargs):
        assert len(parent.shape) == 1
        return np.vectorize(functools.partial(self.mutate))(parent, self.lower, self.upper, self.bits)

    def mutate(self, parent, lower, upper, bits):
        mask = 1
        for i in range(bits):
            if self.random.random() < self.kwargs['prob']:
                parent ^= mask
            mask <<= 1
        return np.clip(parent, lower, upper)
