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

from . import critic


def seed(module):
    class Module(module):
        def __init__(self, *args, **kwargs):
            try:
                torch.manual_seed(kwargs['config']['seed'] + kwargs.get('index', 0))
            except KeyError:
                pass
            super().__init__(*args, **kwargs)
    return Module
