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

import torch.nn as nn

from lamarckian.model import Channel


def share0(module):
    class Module(module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            channel = Channel(self.channel)
            self.critic = nn.Linear(int(channel), len(kwargs['reward']))

        def forward(self, *args, **kwargs):
            outputs = super().forward(*args, **kwargs)
            outputs['critic'] = self.critic(self.share.squeeze(-1).squeeze(-1))
            return outputs
    return Module


def share1(module):
    class Module(module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            channel = Channel(self.channel)
            self.critic = nn.Sequential(
                nn.Linear(int(channel), channel(128)),
                nn.LeakyReLU(),
                nn.Linear(int(channel), len(kwargs['reward'])),
            )

        def forward(self, *args, **kwargs):
            outputs = super().forward(*args, **kwargs)
            outputs['critic'] = self.critic(self.share.squeeze(-1).squeeze(-1))
            return outputs
    return Module


def share2(module):
    class Module(module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            channel = Channel(self.channel)
            self.critic = nn.Sequential(
                nn.Linear(int(channel), channel(128)),
                nn.LeakyReLU(),
                nn.Linear(int(channel), channel(64)),
                nn.LeakyReLU(),
                nn.Linear(int(channel), len(kwargs['reward'])),
            )

        def forward(self, *args, **kwargs):
            outputs = super().forward(*args, **kwargs)
            outputs['critic'] = self.critic(self.share.squeeze(-1).squeeze(-1))
            return outputs
    return Module


def full2(module):
    class Module(module):
        def __init__(self, inputs, *args, **kwargs):
            super().__init__(inputs, *args, **kwargs)
            input, = inputs
            channel = Channel(input['shape'][0])
            self.critic = nn.Sequential(
                nn.Linear(int(channel), channel(128)),
                nn.LeakyReLU(),
                nn.Linear(int(channel), channel(64)),
                nn.LeakyReLU(),
                nn.Linear(int(channel), len(kwargs['reward'])),
            )

        def forward(self, x, *args, **kwargs):
            outputs = super().forward(x, *args, **kwargs)
            outputs['critic'] = self.critic(x)
            return outputs
    return Module
