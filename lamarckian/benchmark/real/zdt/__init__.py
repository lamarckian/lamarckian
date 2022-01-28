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
NAME = os.path.basename(os.path.dirname(__file__))


def distance1(decision):
    return np.mean(decision)


def shape1(x):
    return 1 - np.sqrt(x)


class ZDT(Benchmark):
    def describe(self):
        dist = glom.glom(self.kwargs, f'config.benchmark.{NAME}.decision.dist', default=29)
        encoding = super().describe()
        encoding[CODING] = collections.OrderedDict([('pos', np.array([0, 1], np.float))] + [(f'dist{i}', np.array([0, 1], np.float)) for i in range(dist)])
        return encoding


class ZDT1(ZDT):
    def evaluate(self):
        self.cost += 1
        decision = self.decision[CODING]
        distance = distance1(decision[1:])
        x = decision[0]
        objective = np.array([x, shape1(x / distance)])
        return [dict(fitness=-distance, objective=-objective)]
