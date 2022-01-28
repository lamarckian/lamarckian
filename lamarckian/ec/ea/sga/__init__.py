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

import os
import types
import traceback

import glom

import lamarckian
from lamarckian.ec import Async, wrap as _wrap
from . import wrap

NAME = os.path.basename(os.path.dirname(__file__))


@wrap.elite
class SGA(Async):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mating = lamarckian.ec.mating.Tournament(self.rpc_any('get_choose'), **kwargs)

    def mating(self):
        return self._mating(self.population)

    def selection(self, population):
        return lamarckian.ec.selection.truncate(population, glom.glom(self.kwargs['config'], 'ec.population'))

    def __call__(self):
        offspring = self.breeding(glom.glom(self.kwargs['config'], 'ec.population') * 2)
        self.population = self.selection(offspring)
        return dict(offspring=offspring)


class SGA_(SGA):
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
        self.recorder.register(
            lamarckian.util.counter.Time(**glom.glom(kwargs['config'], 'record.scalar')),
            lambda *args, **kwargs: lamarckian.util.record.Scalar(self.cost, **lamarckian.util.duration.stats),
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
        try:
            self.recorder.put(lamarckian.util.record.Text(self.cost, **{f"topology/{i}": text for i, text in enumerate(self.rpc_all.fetch_all('describe_rpc_all'))}))
        except:
            traceback.print_exc()

    def close(self):
        self.saver()
        self.recorder.close()
        return super().close()

    def __call__(self):
        outcome = super().__call__()
        self.recorder(outcome)
        return outcome


class SteadyState(SGA):
    def __call__(self):
        offspring = self.__next__()
        self.population = self.selection(self.population + offspring)
        return dict(offspring=offspring)


class SteadyState_(SteadyState):
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
        self.recorder.register(
            lamarckian.util.counter.Time(**glom.glom(kwargs['config'], 'record.scalar')),
            lambda *args, **kwargs: lamarckian.util.record.Scalar(self.cost, **lamarckian.util.duration.stats),
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
        self.recorder.put(lamarckian.util.record.Text(self.cost, **{f"topology/{i}": text for i, text in enumerate(self.rpc_all.fetch_all('describe_rpc_all'))}))

    def close(self):
        self.saver()
        self.recorder.close()
        return super().close()

    def __call__(self):
        outcome = super().__call__()
        self.recorder(outcome)
        return outcome
