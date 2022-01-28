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
import itertools
import copy

import torch


def transfer(path):
    try:
        path, multiply = path.split('*')
        multiply = int(multiply)
    except ValueError:
        multiply = 1
    try:
        path, indexes = path.split(':')
        indexes = map(int, indexes.split(','))
    except ValueError:
        indexes = None
    path = os.path.expanduser(os.path.expandvars(path))
    state = torch.load(path, map_location=lambda storage, loc: storage)
    try:
        population = state['population']
    except KeyError:
        population = [dict(decision=state['decision'])]
    if indexes is not None:
        population = [population[index] for index in indexes]
    return list(itertools.chain(*[copy.deepcopy(population) for _ in range(multiply)]))
