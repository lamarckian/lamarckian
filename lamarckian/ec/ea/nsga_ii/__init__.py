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

import os
import itertools
import collections
import types
import copy
import codecs
import traceback

import numpy as np
import glom

import lamarckian
from lamarckian.ec.mating import Tournament
from lamarckian.ec import Async
from . import wrap, record

Item = collections.namedtuple('Item', ['objective', 'individual'])
NAME = os.path.basename(os.path.dirname(__file__))


def assign_crowding_distance(population, density='crowding_distance'):
    for individual in population:
        individual[density] = 0
    items = [Item(individual['result']['objective'], individual) for individual in population]
    for i in range(len(items[0].objective)):
        items.sort(key=lambda item: item.objective[i])
        lower, upper = items[0].objective[i], items[-1].objective[i]
        dist = upper - lower
        assert dist >= 0, dist
        if dist > 0:
            items[0].individual[density] = np.inf
            items[-1].individual[density] = np.inf
            for item1, item, item2 in zip(items[:-2], items[1:-1], items[2:]):
                p1, p2 = item1.objective[i], item2.objective[i]
                assert p1 <= item.objective[i] <= p2
                item.individual[density] += (p2 - p1) / dist
    return population


class Mating(Tournament):
    def __init__(self, choose, **kwargs):
        super().__init__(choose, **kwargs)
        var = {}
        with codecs.open(os.path.join(os.path.dirname(lamarckian.__file__), 'import.py'), 'r', 'utf-8') as f:
            exec(f.read(), var)
        self.dominate = eval('lambda individual1, individual2: ' + glom.glom(kwargs['config'], 'pareto.dominate'), var)
        self.density = eval('lambda individual: ' + glom.glom(kwargs['config'], f'ec.ea.{NAME}.density', default="individual['crowding_distance']"))

    def compete(self, item1, item2):
        index1, individual1 = item1
        index2, individual2 = item2
        if self.dominate(individual1, individual2):
            return item1
        elif self.dominate(individual2, individual1):
            return item2
        else:
            if self.density(individual1) > self.density(individual2):
                return item1
            else:
                return item2


class NSGA_II(Async):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mating = Mating(self.rpc_any('get_choose'), **kwargs)

    def mating(self):
        return self._mating(self.population)

    def evaluation0(self, *args, **kwargs):
        population = super().evaluation0(*args, **kwargs)
        _population = copy.copy(population)
        while _population:
            non_dominated = lamarckian.util.non_dominated.extract(_population, self._mating.dominate)
            assign_crowding_distance(non_dominated)
        return population

    def selection(self, population, size):
        layers = lamarckian.ec.selection.nds(population, size, self._mating.dominate)
        non_critical = layers[:-1]
        for layer in non_critical:
            self.assign_non_critical(layer)
        non_critical = list(itertools.chain(*non_critical))
        assert len(non_critical) < size, (len(non_critical), size)
        _size = size - len(non_critical)
        assert len(layers[-1]) >= _size, (len(layers[-1]), _size)
        critical = self.select_critical(layers[-1], _size)
        assert len(critical) == _size, (len(critical), _size)
        return non_critical + critical

    def assign_non_critical(self, population):
        assign_crowding_distance(population)

    def select_critical(self, population, size):
        assign_crowding_distance(population)
        population.sort(key=self._mating.density, reverse=True)
        return population[:size]

    def __call__(self):
        size = glom.glom(self.kwargs['config'], 'ec.population')
        offspring = self.breeding(size)
        mixed = self.population + offspring
        assert len(mixed) >= size, (len(mixed), size)
        self.population = self.selection(mixed, size)
        assert len(self.population) == size, (len(self.population), size)
        return dict(offspring=offspring)


@wrap.record.nds(NAME)
class NSGA_II_(NSGA_II):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recorder = lamarckian.util.recorder.Recorder.new(**kwargs)
        self.saver = lamarckian.evaluator.Saver(self)
        self.profiler = lamarckian.evaluator.record.Profiler(self.cost, len(self), **kwargs)
        self.recorder.register(
            lamarckian.util.counter.Time(**glom.glom(kwargs['config'], 'record.save')),
            lambda *args, **kwargs: self.saver(),
        )
        self.recorder.register(
            lamarckian.util.counter.Time(**glom.glom(kwargs['config'], 'record.scalar')),
            lambda *args, **kwargs: self.profiler(self.cost),
        )
        encoding = self.describe()
        self.recorder.register(
            lamarckian.util.counter.Time(**glom.glom(kwargs['config'], 'record.histogram')),
            lambda *args, **kwargs: lamarckian.util.record.Histogram(self.cost, **lamarckian.ec.record.get_hparam(encoding, self.population)),
        )
        self.recorder.register(
            lamarckian.util.counter.Time(**glom.glom(kwargs['config'], 'record.histogram')),
            lambda *args, **kwargs: lamarckian.util.record.Histogram(self.cost, **lamarckian.ec.record.get_population(self.population)),
        )
        self.recorder.register(
            lamarckian.util.counter.Time(**glom.glom(kwargs['config'], 'record.scalar')),
            lambda *args, **kwargs: lamarckian.util.record.Scalar(self.cost, **lamarckian.ec.record.get_population_minmax(self.population)),
        )
        self.recorder.register(
            lamarckian.util.counter.Time(**glom.glom(self.kwargs['config'], 'record.plot')),
            lambda *args, **kwargs: record.get_objective(self.cost, self.population),
        )
        self.recorder.register(
            lamarckian.util.counter.Time(**glom.glom(self.kwargs['config'], 'record.plot')),
            lambda outcome: record.Offspring_(f'objective/offspring', self.cost, outcome['offspring'], **kwargs),
        )
        self.recorder.register(
            lamarckian.util.counter.Time(**glom.glom(self.kwargs['config'], 'record.plot')),
            lambda outcome: record.Train('objective/variation/train', self.cost, outcome['offspring'], **kwargs),
        )
        try:
            self.recorder.put(lamarckian.util.record.Text(self.cost, **{f"topology/{i}": text for i, text in enumerate(self.rpc_all.fetch_all('describe_rpc_all'))}))
        except:
            traceback.print_exc()

    def close(self):
        self.saver.close()
        super().close()
        self.recorder.close()

    def __call__(self):
        outcome = super().__call__()
        self.recorder(outcome)
        return outcome
