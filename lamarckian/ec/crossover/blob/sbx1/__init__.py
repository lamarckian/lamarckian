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

import numpy as np
import glom

from .. import Crossover as _Crossover


class Crossover(_Crossover):
    def __init__(self, encoding, **kwargs):
        super().__init__(encoding, **kwargs)
        self.rs = np.random.RandomState(glom.glom(kwargs['config'], 'seed', default=None))
        self.distribution_index = kwargs['sbx']['distribution_index']

    def __call__(self, parent1, parent2, **kwargs):
        return tuple(zip(*[self.make_layer(p1, p2) for p1, p2 in zip(parent1, parent2)]))

    def make_layer(self, parent1, parent2):
        dist = np.linalg.norm(parent1 - parent2)
        if dist > 0:
            offspring = self.crossover(0, dist)
            v = (parent2 - parent1) / dist
            return (parent1 + v * float(d) for d in offspring)
        else:
            return parent1, parent2

    def crossover(self, parent1, parent2):
        assert self.distribution_index >= 0
        u = self.rs.random()
        if u < 0.5:
            spread_factor = (2 * u) ** (1 / (self.distribution_index + 1))
        else:
            spread_factor = (1 / 2 / (1 - u)) ** (1 / (self.distribution_index + 1))
        middle = (parent1 + parent2) / 2
        dist = np.fabs(parent1 - parent2)
        dist2 = dist / 2
        child1 = middle - spread_factor * dist2
        child2 = middle + spread_factor * dist2
        if self.rs.random() < 0.5:
            return child1, child2
        else:
            return child2, child1
