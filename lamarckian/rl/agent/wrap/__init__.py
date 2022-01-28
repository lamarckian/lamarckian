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

import inspect

import torch
import torch.nn as nn


def lstm(agent):
    NAME_FUNC = f'{inspect.getframeinfo(inspect.currentframe()).function}'
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    class Agent(agent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            try:
                lstm = next((module for module in self.model.modules() if isinstance(module, nn.LSTM)))
                setattr(self, PATH_FUNC, (
                    torch.zeros(1, lstm.num_layers, lstm.hidden_size).to(self.device),
                    torch.zeros(1, lstm.num_layers, lstm.hidden_size).to(self.device),
                ))
            except StopIteration:
                setattr(self, PATH_FUNC, tuple())

        def get_inputs(self, *args, **kwargs):
            return super().get_inputs(*args, **kwargs) + getattr(self, PATH_FUNC)

        def forward(self, *args, **kwargs):
            outputs = super().forward(*args, **kwargs)
            if getattr(self, PATH_FUNC):
                setattr(self, PATH_FUNC, outputs['hidden'])
            return outputs
    return Agent
