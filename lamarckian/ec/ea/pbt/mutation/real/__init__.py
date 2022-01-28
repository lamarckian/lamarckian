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

from lamarckian.ec.mutation.real import Mutation
from . import wrap


class Perturb(Mutation):
    def __init__(self, encoding, **kwargs):
        super().__init__(encoding, **kwargs)
        self.random = random.Random(glom.glom(kwargs['config'], 'seed', default=None))
        self.perturb = glom.glom(kwargs['config'], 'ec.mutation.real.perturb')

    def __call__(self, parent, **kwargs):
        assert len(parent.shape) == 1, parent.shape
        return np.vectorize(lambda parent, lower, upper: np.clip(self.random.choice(self.perturb) * parent, lower, upper))(parent, self.lower, self.upper)
