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
import glom


def get_ratio(self, outcome, tag):
    try:
        clip = self.hparam['clip']
    except (AttributeError, KeyError):
        clip = glom.glom(self.kwargs['config'], 'rl.ppo.clip')
    ratio = outcome['ratio'].detach()
    return {
        f'{tag}/ratio/min': ratio.min().item(),
        f'{tag}/ratio/max': ratio.max().item(),
        f'{tag}/ratio/clip': torch.logical_or(ratio < 1 - clip, ratio > 1 + clip).int().sum().item() / np.multiply.reduce(ratio.shape),
    }
