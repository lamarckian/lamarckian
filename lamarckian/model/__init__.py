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

import collections

import numpy as np
import torch


class Channel(object):
    def __init__(self, channel):
        self.channel = channel

    def __call__(self):
        return self.channel

    def set(self, channel):
        self.channel = channel

    def next(self, channel):
        self.channel = channel
        return self.channel


def to_blob(state_dict):
    return [value.cpu().numpy() for value in state_dict.values()]


def from_blob(blob, keys, device=torch.device('cpu')):
    return collections.OrderedDict([(key, torch.from_numpy(np.array(value)).to(device)) for key, value in zip(keys, blob)])


from . import wrap, group
