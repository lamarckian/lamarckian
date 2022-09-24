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


def linear(decision_pos, num):
    num_pos = len(decision_pos)
    decision_ext = np.hstack([decision_pos, [0]])
    return np.array([np.prod(decision_pos[:num_pos - i]) * (1 - decision_ext[num_pos - i]) for i in range(num)])


def concave(decision_pos, num):
    num_pos = len(decision_pos)
    angle = decision_pos * np.pi / 2
    cos = np.cos(angle)
    sin = np.hstack([np.sin(angle), [1]])
    return np.array([np.prod(cos[:num_pos - i]) * sin[num_pos - i] for i in range(num)])
