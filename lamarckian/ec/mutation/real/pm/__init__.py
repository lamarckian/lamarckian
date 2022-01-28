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
import math
import random
import functools

import numpy as np
import glom

from .. import Mutation as _Mutation
NAME = os.path.basename(os.path.dirname(__file__))


def calc_perturbance_factor_probability(perturbance_factor, distribution_index):
    return (distribution_index + 1) * np.power(1 - np.fabs(perturbance_factor), distribution_index) / 2


def calc_amplification_factor_lower(perturbance_factor_lower, distribution_index):
    assert np.all(-1 <= perturbance_factor_lower) and np.all(perturbance_factor_lower <= 0), perturbance_factor_lower
    assert distribution_index >= 0, distribution_index
    return 2 / (1 - np.power(1 + perturbance_factor_lower, distribution_index + 1))


def calc_amplification_factor_upper(perturbance_factor_upper, distribution_index):
    assert np.all(0 <= perturbance_factor_upper) and np.all(perturbance_factor_upper <= 1), perturbance_factor_upper
    assert distribution_index >= 0, distribution_index
    return 2 / (1 - np.power(1 - perturbance_factor_upper, distribution_index + 1))


def generate_perturbance_factor_instance_lower(amplification_factor_lower, distribution_index, u):
    assert amplification_factor_lower > 1, amplification_factor_lower
    return math.pow(2 * u + (1 - 2 * u) * (1 - 2 / amplification_factor_lower), 1. / (distribution_index + 1)) - 1


def generate_perturbance_factor_instance_upper(amplification_factor_upper, distribution_index, u):
    assert amplification_factor_upper > 1, amplification_factor_upper
    return 1 - math.pow(2 * (1 - u) + (2 * u - 1) * (1 - 2 / amplification_factor_upper), 1. / (distribution_index + 1))


def _calc_amplification_factor_lower(perturbance_factor_lower, distribution_index):
    assert -1 <= perturbance_factor_lower <= 0, perturbance_factor_lower
    assert distribution_index >= 0, distribution_index
    return math.pow(1 + perturbance_factor_lower, distribution_index + 1)


def _calc_amplification_factor_upper(perturbance_factor_upper, distribution_index):
    assert 0 <= perturbance_factor_upper <= 1, perturbance_factor_upper
    assert distribution_index >= 0, distribution_index
    return math.pow(1 - perturbance_factor_upper, distribution_index + 1)


def _generate_perturbance_factor_instance_lower(_amplification_factor_lower, distribution_index, u):
    return math.pow(2 * u + (1 - 2 * u) * _amplification_factor_lower, 1. / (distribution_index + 1)) - 1


def _generate_perturbance_factor_instance_upper(_amplification_factor_upper, distribution_index, u):
    return 1 - math.pow(2 * (1 - u) + (2 * u - 1) * _amplification_factor_upper, 1. / (distribution_index + 1))


class Mutation(_Mutation):
    def __init__(self, encoding, **kwargs):
        super().__init__(encoding, **kwargs)
        self.random = random.Random(glom.glom(kwargs['config'], 'seed', default=None))
        self.distribution_index = kwargs[NAME]['distribution_index']

    def __call__(self, parent, **kwargs):
        assert len(parent.shape) == 1, parent.shape
        return np.vectorize(functools.partial(self.mutate))(parent, self.lower, self.upper)

    def mutate(self, parent, lower, upper):
        if self.random.random() >= self.kwargs['prob']:
            return parent
        assert lower <= parent <= upper, (lower, parent, upper)
        distance = upper - lower
        assert distance > 0, distance
        u = self.random.random()
        if u < 0.5:
            # Lower
            perturbance_factor_lower = (lower - parent) / distance
            _amplification_factor_lower = _calc_amplification_factor_lower(perturbance_factor_lower, self.distribution_index)
            perturbance_factor = _generate_perturbance_factor_instance_lower(_amplification_factor_lower, self.distribution_index, u)
            assert perturbance_factor <= 0, perturbance_factor
        else:
            # Upper
            perturbance_factor_upper = (upper - parent) / distance
            _amplification_factor_upper = _calc_amplification_factor_upper(perturbance_factor_upper, self.distribution_index)
            perturbance_factor = _generate_perturbance_factor_instance_upper(_amplification_factor_upper, self.distribution_index, u)
            assert 0 <= perturbance_factor, perturbance_factor
        child = parent + perturbance_factor * distance
        child = np.clip(child, lower, upper)
        return child
