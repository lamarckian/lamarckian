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

from .. import Benchmark, CODING
from . import distance, shape, transform
NAME = os.path.basename(os.path.dirname(__file__))


class DTLZ(Benchmark):
    def describe(self):
        self.objective = glom.glom(self.kwargs, f'config.benchmark.{NAME}.objective')
        self.num_pos = self.objective - 1
        dist = glom.glom(self.kwargs, f'config.benchmark.{NAME}.decision.dist')
        encoding = super().describe()
        encoding[CODING] = collections.OrderedDict([(f'pos{i}', np.array([0, 1], np.float)) for i in range(self.num_pos)] + [(f'dist{i}', np.array([0, 1], np.float)) for i in range(dist)])
        return encoding


class DTLZ1(DTLZ):
    def evaluate(self):
        self.cost += 1
        decision = self.decision[CODING]
        g = distance.g1(decision[self.num_pos:])
        objective = shape.linear(decision[:self.num_pos], self.objective) * (1 + g) / 2
        return [dict(fitness=-g, objective=-objective)]


class DTLZ2(DTLZ):
    def evaluate(self):
        self.cost += 1
        decision = self.decision[CODING]
        g = distance.g2(decision[self.num_pos:])
        objective = shape.concave(decision[:self.num_pos], self.objective) * (1 + g)
        return [dict(fitness=-g, objective=-objective)]


class DTLZ3(DTLZ):
    def evaluate(self):
        self.cost += 1
        decision = self.decision[CODING]
        g = distance.g1(decision[self.num_pos:])
        objective = shape.concave(decision[:self.num_pos], self.objective) * (1 + g)
        return [dict(fitness=-g, objective=-objective)]


class DTLZ4(DTLZ):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = glom.glom(kwargs, f'config.benchmark.{NAME}.alpha', default=100)

    def evaluate(self):
        self.cost += 1
        decision = self.decision[CODING]
        g = distance.g2(decision[self.num_pos:])
        decision_pos = transform.bias.polynomial(decision[:self.num_pos], self.alpha)
        objective = shape.concave(decision_pos, self.objective) * (1 + g)
        return [dict(fitness=-g, objective=-objective)]


class DTLZ5(DTLZ):
    def evaluate(self):
        self.cost += 1
        decision = self.decision[CODING]
        g = distance.g2(decision[self.num_pos:])
        decision_pos = transform.degenerate(decision[:self.num_pos], g, 1)
        objective = shape.concave(decision_pos, self.objective) * (1 + g)
        return [dict(fitness=-g, objective=-objective)]


class DTLZ6(DTLZ):
    def evaluate(self):
        self.cost += 1
        decision = self.decision[CODING]
        g = distance.g3(decision[self.num_pos:])
        decision_pos = transform.degenerate(decision[:self.num_pos], g, 1)
        objective = shape.concave(decision_pos, self.objective) * (1 + g)
        return [dict(fitness=-g, objective=-objective)]


def h1(decision, num, g):
    assert num > 0, num
    return num - sum([(np.sin(3 * np.pi * decision[obj]) + 1) * decision[obj] / (g + 1) for obj in range(num)])


class DTLZ7(DTLZ):
    def evaluate(self):
        self.cost += 1
        decision = self.decision[CODING]
        g = distance.g4(decision[self.num_pos:])
        objective = np.hstack([decision[:self.num_pos], [(g + 1) * h1(decision, self.objective, g)]])
        return [dict(fitness=-g, objective=-objective)]
