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
import collections

import numpy as np
import glom

from .. import Benchmark as _Benchmark
CODING = os.path.basename(os.path.dirname(__file__))


class Benchmark(_Benchmark):
    def initialize(self):
        encoding = self.describe()[CODING]
        if isinstance(encoding, collections.abc.Mapping):
            lower, upper = np.stack(list(encoding.values())).T
        elif isinstance(encoding, np.ndarray):
            lower, upper = encoding.T
        else:
            raise TypeError(type(encoding))
        decision = super().initialize()
        decision[CODING] = lower + np.random.random(lower.shape) * (upper - lower)
        return decision


from . import zdt, dtlz


class Sphere(Benchmark):
    def describe(self):
        decision = glom.glom(self.kwargs, 'config.benchmark.sphere.decision', default=2)
        encoding = super().describe()
        encoding[CODING] = np.array([(-1, 1) for _ in range(decision)], np.float)
        return encoding

    def evaluate(self):
        self.cost += 1
        decision = self.decision[CODING]
        return [dict(fitness=-np.sum(decision ** 2))]


class Square(Benchmark):
    def describe(self):
        self.optimal = glom.glom(self.kwargs, 'config.benchmark.square.optimal', default=0.5)
        encoding = super().describe()
        encoding[CODING] = np.array([(self.optimal - 1, self.optimal + 1)], np.float)
        return encoding

    def evaluate(self):
        self.cost += 1
        x, = self.decision[CODING]
        return [dict(fitness=-np.sum(np.square(x - self.optimal), -1))]


class XSinX(Benchmark):
    def describe(self):
        encoding = super().describe()
        encoding[CODING] = np.array([(0, 3)], np.float)
        return encoding

    def evaluate(self):
        self.cost += 1
        x, = self.decision[CODING]
        return [dict(fitness=np.sum(x * np.sin(10 * np.pi * x) + 2, -1))]


class Shubert(Benchmark):
    def describe(self):
        encoding = super().describe()
        encoding[CODING] = np.array([(-10, 10), (-10, 10)], np.float)
        return encoding

    def evaluate(self):
        self.cost += 1
        x1, x2 = self.decision[CODING]
        part1 = sum(i * np.cos((i + 1) * x1 + i) for i in range(1, 6))
        part2 = sum(i * np.cos((i + 1) * x2 + i) for i in range(1, 6))
        fitness = -part1 * part2
        return [dict(fitness=fitness)]


class Rosenbrock(Benchmark):
    def describe(self):
        encoding = super().describe()
        self.alpha = glom.glom(self.kwargs, 'config.benchmark.rosenbrock.alpha', default=1e2)
        encoding[CODING] = np.array([(-10, 10), (-10, 10)], np.float)
        return encoding

    def evaluate(self):
        self.cost += 1
        decision = self.decision[CODING]
        fitness = sum(self.alpha * (decision[:-1] ** 2 - decision[1:]) ** 2 + (1. - decision[:-1]) ** 2)
        return [dict(fitness=fitness)]
