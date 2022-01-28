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

import lamarckian


def test():
    def shaping(i, potential, reward, done, discount):
        if done:
            value = 0
        else:
            value = potential[i] + reward
        reward = discount * value - potential[i]
        potential[i] = value
        return reward
    np.random.seed(0)
    reward = np.random.random(100)
    potential = np.zeros(1)
    discount = 0.99
    reward_ = np.array([shaping(0, potential, r, done, discount) for r, done in zip(reward, [False] * (len(reward) - 1) + [True])])
    credit, credit_ = lamarckian.rl.cumulate(reward, discount)[0], lamarckian.rl.cumulate(reward_, discount)[0]
    assert abs(credit) > 1, credit
    assert abs(credit_) < np.finfo(float).eps, credit_
