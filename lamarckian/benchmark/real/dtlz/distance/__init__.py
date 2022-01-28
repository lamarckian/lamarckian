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

import numpy as np


def g1(decision_dist):
    tmp = decision_dist - 0.5
    return 100 * (len(decision_dist) + np.sum(np.square(tmp) - np.cos(20 * np.pi * tmp)))


def g2(decision_dist):
    return sum(np.square(decision_dist - 0.5))


def g3(decision_dist):
    return sum(np.power(decision_dist, .1))


def g4(decision_dist):
    return 1 + 9 * sum(decision_dist) / len(decision_dist)
