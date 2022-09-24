"""
Copyright (C) 2020, 柏卉 (Hui Bai)

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

import copy
import itertools
import random

import glom

import lamarckian
from lamarckian.ec import Sync
from lamarckian.ec.ea import sga
from . import wrap


@sga.wrap.elite
class CSO(Sync):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.random = random.Random(glom.glom(kwargs['config'], 'seed', default=None))

    def evaluation0(self, *args, **kwargs):
        population = super().evaluation0(*args, **kwargs)
        assert len(population) % 2 == 0, len(population)
        self.keys = set(population[0]['cso']['velocity'])
        return population

    def get_center(self):
        return {}

    def compete(self, individual1, individual2):
        if individual1['result']['fitness'] > individual2['result']['fitness']:
            return individual1, individual2
        else:
            return individual2, individual1

    def __call__(self):
        center = self.get_center()
        assert set(center) == self.keys, center.keys()
        population = copy.copy(self.population)
        self.random.shuffle(population)
        winners, losers = zip(*[self.compete(*parent) for parent in itertools.zip_longest(*(iter(population),) * 2)])
        for winner, loser in zip(winners, losers):
            loser['cso']['winner'] = {key: value for key, value in winner['decision'].items() if key in self.keys}
            loser['cso']['center'] = center
        losers = list(itertools.chain(*self.rpc_any.map((('variation', (), {}) for loser in losers))))
        self.cost += sum(sum(individual['cost'].values()) for individual in losers)
        self.population = list(winners) + losers
        return dict(center=center)


class Swarm(CSO):
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

    def close(self):
        self.saver.close()
        super().close()
        self.recorder.close()

    def __call__(self):
        outcome = super().__call__()
        self.recorder(outcome)
        return outcome
