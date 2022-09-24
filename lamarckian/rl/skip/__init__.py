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


def pad(skip, *args, dim=0):
    tensors = []
    for trajectory in args:
        trajectory = torch.unbind(trajectory, dim)
        assert len(trajectory) == len(skip), (len(trajectory), len(skip))
        padded = []
        for exp, n in zip(trajectory, skip):
            padded.append(exp)
            padded += [torch.zeros_like(exp).to(exp.device)] * n
        tensors.append(torch.stack(padded, dim))
    return tensors


def restore(skip, *args, dim=0):
    tensors = []
    for padded in args:
        padded = torch.unbind(padded, dim)
        assert len(padded) == len(skip) + skip.sum().item(), (len(padded), len(skip) + skip.sum().item())
        trajectory = []
        for n in skip:
            trajectory.append(padded[0])
            padded = padded[n + 1:]
        assert len(trajectory) == len(skip), (len(trajectory), len(skip))
        tensors.append(torch.stack(trajectory, dim))
    return tensors
