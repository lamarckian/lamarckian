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

import torch.nn as nn

from lamarckian.model import Channel


class Module(nn.Module):
    def __init__(self, inputs, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        channel = Channel(inputs)
        self.linear = nn.Sequential(
            nn.Linear(channel(), channel.next(8)),
            nn.Tanh(),
            nn.Linear(channel(), 1),
        )

    def forward(self, x):
        return self.linear(x)
