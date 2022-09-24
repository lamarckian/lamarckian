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


def extract(population, dominate):
    assert population
    non_dominated = [population.pop()]
    for i, individual in reversed(list(enumerate(population))):
        assert non_dominated
        flag = True
        for e, elite in reversed(list(enumerate(non_dominated))):
            if dominate(individual, elite):
                population.append(non_dominated.pop(e))
            elif dominate(elite, individual):
                flag = False
                break
        if flag:
            non_dominated.append(population.pop(i))
    return non_dominated
