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

import inspect
import random

import numpy as np
import glom

import lamarckian
from . import real, integer, blob, wrap


class Crossover(object):
    def __init__(self, encoding, **kwargs):
        config = glom.glom(kwargs['config'], 'ec.crossover')
        self.crossover = {}
        for key, encoding in encoding.items():
            _kwargs = {**kwargs, **config[key]}
            cls = lamarckian.evaluator.parse(*_kwargs['create'])
            self.crossover[key] = cls(encoding, **_kwargs)
        self._choose = {key: len(inspect.getargspec(crossover.__call__).args) - 1 for key, crossover in self.crossover.items()}
        self.choose = max(*self._choose.values(), 2)
        for key, value in self._choose.items():
            if value <= 0:
                self._choose[key] = self.choose
        assert min(self._choose.values()) > 0 and max(self._choose.values()) > 1, self._choose
        self.random = random.Random(glom.glom(kwargs['config'], 'seed', default=None))

    def close(self):
        for crossover in self.crossover.values():
            crossover.close()

    def __call__(self, ancestor):
        assert len(ancestor) == self.choose, (len(ancestor), self.choose)
        _offspring = {}
        for key, crossover in self.crossover.items():
            choose = self._choose[key]
            _ancestor = ancestor[:choose]
            decisions = [individual['decision'][key] for individual in _ancestor]
            _offspring[key] = crossover(*decisions, ancestor=_ancestor)
        offspring = [dict(decision={}) for _ in range(min(map(len, _offspring.values())))]
        for key, decisions in _offspring.items():
            assert len(decisions) >= len(offspring), (len(decisions), len(offspring))
            decisions = list(decisions)
            self.random.shuffle(decisions)
            for decision, child in zip(decisions, offspring):
                child['decision'][key] = decision
        return offspring


class Useless(object):
    def __init__(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def __call__(self, *ancestor, **kwargs):
        return ancestor


class Best(object):
    def __init__(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def __call__(self, *decisions, **kwargs):
        assert len(decisions) > 1, len(decisions)
        assert len(decisions) == len(kwargs['ancestor']), (len(decisions), len(kwargs['ancestor']))
        index = np.argmax([parent['result']['fitness'] for parent in kwargs['ancestor']])
        return decisions[index],
