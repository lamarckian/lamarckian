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


def dominate_min(p1, p2):
    assert len(p1) == len(p2), (len(p1), len(p2))
    dominated = False
    for x1, x2 in zip(p1, p2):
        if x2 < x1:
            return False
        if x1 < x2:
            dominated = True
    return dominated


def dominate_max(p1, p2):
    assert len(p1) == len(p2), (len(p1), len(p2))
    dominated = False
    for x1, x2 in zip(p1, p2):
        if x2 > x1:
            return False
        if x1 > x2:
            dominated = True
    return dominated
