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

import torch
import torch.nn as nn
import torch.nn.functional as F
import glom

from .. import Channel


class Module(nn.Module):
    def __init__(self, inputs, **kwargs):
        super().__init__()
        input, = inputs
        self.kwargs = kwargs
        channel = Channel(input['shape'][0])
        self.linear = nn.Sequential(
            nn.Linear(channel(), channel.next(512)),
            nn.LeakyReLU(),
            nn.Linear(channel(), channel.next(256)),
            nn.LeakyReLU(),
        )
        self.channel = channel()
        if glom.glom(kwargs['config'], 'model.lstm', default=False):
            self.lstm = nn.LSTM(channel(), channel(), batch_first=True)
        self.discrete = nn.ModuleList([nn.Sequential(
            nn.Linear(channel(), channel.next(128)),
            nn.LeakyReLU(),
            nn.Linear(channel(), len(names)),
        ) for names in kwargs.get('discrete', [])])

    def forward(self, x, *hidden):
        self.share = self.linear(x)
        outputs = dict(
            discrete=[output(self.share) for output in self.discrete],
        )
        if hasattr(self, 'lstm'):
            batch_size, dim = x.shape
            share, hidden = self.lstm(self.share.view(batch_size, 1, -1), tuple(t.swapaxes(0, 1) for t in hidden))
            self.share = share.view(batch_size, -1)
            outputs['hidden'] = tuple(t.swapaxes(0, 1) for t in hidden)
        return outputs
