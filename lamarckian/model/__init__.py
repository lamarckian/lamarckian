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

import collections

import numpy as np
import torch


class Channel(object):
    def __init__(self, value):
        self.value = value

    def __call__(self, value):
        self.value = value
        return self.value

    def __int__(self):
        return int(self.value)

    def __repr__(self):
        return f"{self.value}"


def to_blob(state_dict):
    return [value.cpu().numpy() for value in state_dict.values()]


def from_blob(blob, keys, device=torch.device('cpu')):
    return collections.OrderedDict([(key, torch.from_numpy(np.array(value)).to(device)) for key, value in zip(keys, blob)])


from . import wrap, group
