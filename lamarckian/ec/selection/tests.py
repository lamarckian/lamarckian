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

from . import nds


def test_nds():
    from lamarckian.util.pareto import dominate_min as dominate
    for n, dimension in [(200, 2), (300, 3)]:
        require = n / 2
        population = [np.random.rand(dimension) for _ in range(n)]
        layers = nds(population, require, dominate)
        assert(len(layers) > 0)
        non_critical = sum([len(layer) for layer in layers[:-1]])
        candidate = non_critical + len(layers[-1])
        assert(non_critical < require <= candidate)
        assert(candidate <= n)
        assert(candidate + len(population) == n)
