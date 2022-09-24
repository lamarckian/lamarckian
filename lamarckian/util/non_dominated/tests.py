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

from . import extract


def generate_random(n, dimension):
    assert(n > 0)
    assert(dimension > 0)
    population = []
    for _ in range(n):
        population.append(np.random.rand(dimension) - .5)
    return population


def check_elite(elite, population, dominate):
    assert not dominate(elite, elite)
    for individual in population:
        if dominate(individual, elite):
            return False
    return True


def check_non_dominated(non_dominated, population, dominate):
    for elite in non_dominated:
        if not check_elite(elite, population, dominate):
            return False
    return True


def test(n=300):
    from lamarckian.util.pareto import dominate_min as dominance
    for dimension in range(2, 21):
        for _ in range(5):
            population = generate_random(n, dimension)
            non_dominated = extract(population, dominance)
            assert len(non_dominated) + len(population) == n
            assert check_non_dominated(non_dominated, non_dominated, dominance)
            assert check_non_dominated(non_dominated, population, dominance)
