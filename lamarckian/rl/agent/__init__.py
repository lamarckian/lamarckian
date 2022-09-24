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

import torch

from . import wrap


class Agent(object):
    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.device = next(model.parameters()).device

    def close(self):
        pass

    def get_inputs(self, state):
        return tuple(torch.FloatTensor(a).unsqueeze(0).to(self.device) for a in state['inputs'])

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        return state
