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

import random

import numpy as np
import glom

from ... import wrap as _wrap
from .. import Crossover as _Crossover


@_wrap.group
class Split(_Crossover):
    def __init__(self, encoding, **kwargs):
        super().__init__(encoding, **kwargs)
        self.random = random.Random(glom.glom(kwargs['config'], 'seed', default=None))

    def __call__(self, parent1, parent2, **kwargs):
        point = self.random.choice(self.group + [None])
        child1 = []
        child2 = []
        for group in self.group:
            if group == point:
                parent1, parent2 = parent2, parent1
            for i in group:
                child1.append(parent1[i])
                child2.append(parent2[i])
        return child1, child2


@_wrap.group
class Split1(_Crossover):
    def __init__(self, encoding, **kwargs):
        super().__init__(encoding, **kwargs)
        self.random = random.Random(glom.glom(kwargs['config'], 'seed', default=None))

    def __call__(self, parent1, parent2, **kwargs):
        point = self.random.choice(self.group + [None])
        parent = parent1
        child = []
        for group in self.group:
            if group == point:
                parent = parent2
            for i in group:
                child.append(parent[i])
        return child,


@_wrap.group
class SplitLayerAny(_Crossover):
    def __init__(self, encoding, **kwargs):
        super().__init__(encoding, **kwargs)
        self.random = random.Random(glom.glom(kwargs['config'], 'seed', default=None))

    def __call__(self, parent1, parent2, **kwargs):
        point = self.random.choice(self.group)
        child1 = []
        child2 = []
        for group in self.group:
            if group == point:
                _key = group[0]
                channels = parent1[_key].shape[0]
                j = self.random.randint(0, channels)
                for i in group:
                    assert parent1[i].shape == parent2[i].shape
                    if parent1[i].shape:
                        assert parent1[i].shape[0] == channels
                        child1.append(np.concatenate([parent1[i][:j], parent2[i][j:]]))
                        child2.append(np.concatenate([parent2[i][:j], parent1[i][j:]]))
                parent1, parent2 = parent2, parent1
            else:
                for i in group:
                    child1.append(parent1[i])
                    child2.append(parent2[i])
        return child1, child2


@_wrap.group
class SplitLayerAny1(_Crossover):
    def __init__(self, encoding, **kwargs):
        super().__init__(encoding, **kwargs)
        self.random = random.Random(glom.glom(kwargs['config'], 'seed', default=None))

    def __call__(self, parent1, parent2, **kwargs):
        point = self.random.choice(self.group)
        parent = parent1
        child = []
        for group in self.group:
            if group == point:
                _key = group[0]
                channels = parent1[_key].shape[0]
                j = self.random.randint(0, channels)
                for i in group:
                    assert parent1[i].shape == parent2[i].shape
                    if parent1[i].shape:
                        assert parent1[i].shape[0] == channels
                        child.append(np.concatenate([parent1[i][:j], parent2[i][j:]]))
                parent = parent2
            else:
                for i in group:
                    child.append(parent[i])
        return child,


@_wrap.group
class SplitLayerAll(_Crossover):
    def __init__(self, encoding, **kwargs):
        super().__init__(encoding, **kwargs)
        self.random = random.Random(glom.glom(kwargs['config'], 'seed', default=None))

    def __call__(self, parent1, parent2, **kwargs):
        child1 = []
        child2 = []
        for group in self.group:
            channels = parent1[group[0]].shape[0]
            j = self.random.randint(0, channels)
            for i in group:
                assert parent1[i].shape == parent2[i].shape
                if parent1[i].shape:
                    assert parent1[i].shape[0] == channels
                    child1.append(np.concatenate([parent1[i][:j], parent2[i][j:]]))
                    child2.append(np.concatenate([parent2[i][:j], parent1[i][j:]]))
        return child1, child2
