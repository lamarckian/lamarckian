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

from lamarckian.model import Channel


def hidden1(module):
    @wrap.group.rsplit(1)
    class Module(nn.Module):
        def __init__(self, config, inputs, outputs, *args, **kwargs):
            super().__init__()
            self.config = config
            self.actor = module(config, inputs, outputs, *args, **kwargs)

            class Critic(nn.Module):
                def __init__(self, config, inputs, outputs):
                    super().__init__()
                    self.config = config
                    channel = Channel(inputs)
                    self.linear = nn.Sequential(
                        nn.Linear(channel(), channel.next(128)),
                        nn.LeakyReLU(),
                    )
                    channel.channels += outputs
                    self.value = nn.Sequential(
                        nn.Linear(channel(), channel.next(64)),
                        nn.LeakyReLU(),
                        nn.Linear(channel(), 1),
                    )

                def forward(self, action, *inputs):
                    feature = torch.cat([self.linear(*inputs), action], 1)
                    return self.value(feature)

            self.critic = Critic(config, inputs, outputs, *args, **kwargs)
            for m in self.modules():
                self.init(m)
    return Module
