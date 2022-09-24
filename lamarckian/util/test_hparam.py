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

from . import Hparam


def test():
    tune = Hparam()
    tune.setup('a', 0)
    tune.setup('b', np.array([0, 2], np.float))
    tune.setup('c', np.array([0, 3], np.float))
    tune.setup('d', np.array([1, 2], np.int))
    tune.setup('e', np.array([1, 3], np.int))
    encoding = tune.describe()
    decision = tune.initialize()
    tune.set(decision)
    tune.get()


def test_real():
    tune = Hparam()
    tune.setup('a', 0)
    tune.setup('b', np.array([0, 2], np.float))
    tune.setup('c', np.array([0, 3], np.float))
    tune.describe()
    decision = tune.initialize()
    tune.set(decision)
    tune.get()


def test_integer():
    tune = Hparam()
    tune.setup('a', 0)
    tune.setup('d', np.array([1, 2], np.int))
    tune.setup('e', np.array([1, 3], np.int))
    tune.describe()
    decision = tune.initialize()
    tune.set(decision)
    tune.get()
