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

import time
import ast
import random
import collections

import numpy as np

from . import config, counter, file, pareto, non_dominated, parse, rnd, remote, duration, restoring, testing, record, recorder, mpl, ray_fake, rpc, hook


class Closing(object):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def close(self):
        for value in self.args:
            value.close()
        for key, value in self.kwargs.items():
            value.close()


class Frac(object):
    def __init__(self, count=0, total=0):
        assert 0 <= count <= total, (count, total)
        self.count = count
        self.total = total

    def reset(self):
        self.count = 0
        self.total = 0

    def __add__(self, value):
        self.count += value
        self.total += 1

    def __repr__(self):
        return f'{self.count}/{self.total}'

    def __float__(self):
        return self.count / (self.total + np.finfo(np.float).eps)


class Hparam(object):
    def __init__(self):
        self.value = collections.OrderedDict()
        self.boundary = collections.OrderedDict()

    @staticmethod
    def translate(dtype):
        if np.issubdtype(dtype, np.float):
            return 'real'
        elif np.issubdtype(dtype, np.integer):
            return 'integer'

    def setup(self, key, item, dtype=None, get=np.max):
        assert key not in self.value, (key, list(self.value.keys()))
        if np.isscalar(item):
            value = item
        else:
            lower, upper = item
            assert lower < upper, (key, lower, upper)
            boundary = np.array([lower, upper], dtype)
            coding = type(self).translate(boundary.dtype)
            if coding not in self.boundary:
                self.boundary[coding] = {}
            assert key not in self.boundary[coding], (key, coding)
            self.boundary[coding][key] = boundary
            value = get(boundary)
        self.value[key] = value

    def describe(self):
        return self.boundary

    def initialize_real(self, lower, upper):
        return lower + np.random.random(lower.shape) * (upper - lower)

    def initialize_integer(self, lower, upper):
        return np.array([random.randint(l, u) for l, u in zip(lower, upper)])

    def initialize(self):
        return {coding: getattr(self, f'initialize_{coding}')(*np.stack(list(boundary.values())).T) for coding, boundary in self.boundary.items()}

    def set(self, decision):
        for coding, boundary in self.boundary.items():
            vector = decision[coding]
            assert len(boundary) == len(vector), (boundary.keys(), len(vector))
            for key, value in zip(boundary, vector):
                assert key in self.value, key
                lower, upper = boundary[key]
                assert lower <= value <= upper, (lower, value, upper)
                self.value[key] = value
        return decision

    def get(self):
        return {coding: np.array([self.value[key] for key in boundary]) for coding, boundary in self.boundary.items()}

    def __iter__(self):
        return iter(self.boundary)

    def __getitem__(self, key):
        return self.value[key]

    def __setitem__(self, key, value):
        try:
            lower, upper = self.boundary[key]
            assert lower <= value <= upper, (lower, value, upper)
        except KeyError:
            pass
        self.value[key] = value

    def __setstate__(self, state):
        for key, value in state.get('value', {}).items():
            self.value[key] = value
        for key, boundary in state.get('boundary', {}).items():
            self.boundary[key] = boundary

    def __getstate__(self):
        return dict(value=self.value, boundary=self.boundary)


def reduce(results):
    if len(results) > 1:
        result = {}
        for key, value in results[0].items():
            try:
                result[key] = np.nanmean([result[key] for result in results], 0)
            except:
                pass
        return result
    else:
        return results[0]


def to_probs(counts):
    total = counts.sum()
    if total:
        return counts / total
    else:
        return np.ones_like(counts) / len(counts)


def try_cast(s):
    try:
        return ast.literal_eval(s)
    except:
        try:
            return eval(s)
        except:
            return s


def abs_mean(data, dtype=np.float32):
    assert isinstance(data, np.ndarray), type(data)
    return np.sum(np.abs(data)) / dtype(data.size)
