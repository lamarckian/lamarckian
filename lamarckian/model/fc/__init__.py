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
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import Channel


class Module(nn.Module):
    def __init__(self, inputs, **kwargs):
        super().__init__()
        input, = inputs
        self.kwargs = kwargs
        channel = Channel(input['shape'][0])
        self.linear = nn.Sequential(
            nn.Linear(channel(), channel.next(256)),
            nn.LeakyReLU(),
            nn.Linear(channel(), channel.next(128)),
            nn.LeakyReLU(),
        )
        self.channel = channel()
        self.discrete = nn.ModuleList([nn.Sequential(
            nn.Linear(channel(), channel.next(128)),
            nn.LeakyReLU(),
            nn.Linear(channel(), len(names)),
        ) for names in kwargs.get('discrete', [])])
        if 'continuous' in kwargs:
            lower, upper = kwargs['continuous'].T.copy()
            self.continuous = nn.Sequential(
                nn.Linear(channel(), channel.next(128)),
                nn.LeakyReLU(),
                nn.Linear(channel(), len(lower) * 2),
                nn.Unflatten(-1, [len(lower), 2]),
            )
            self.continuous.lower = torch.FloatTensor(lower).unsqueeze(0)
            self.continuous.range = torch.FloatTensor(upper - lower).unsqueeze(0)

    def to(self, device, *args, **kwargs):
        if hasattr(self, 'continuous'):
            self.continuous.lower = self.continuous.lower.to(device, *args, **kwargs)
            self.continuous.range = self.continuous.range.to(device, *args, **kwargs)
        return super().to(device, *args, **kwargs)

    def forward(self, x):
        self.share = self.linear(x)
        outputs = dict(
            discrete=[output(self.share) for output in self.discrete],
        )
        if hasattr(self, 'continuous'):
            mean, var = torch.unbind(self.continuous(self.share), -1)
            mean = torch.tanh(mean) + 1 / 2
            mean = self.continuous.lower + mean * self.continuous.range
            var = F.softplus(var) + np.finfo(np.float32).eps
            outputs['continuous'] = (mean, var)
        return outputs
