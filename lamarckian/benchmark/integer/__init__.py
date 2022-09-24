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

import os
import collections
import random

import numpy as np

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
        decision[CODING] = np.array([random.randint(l, u) for l, u in zip(lower, upper)])
        return decision
