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
import threading
import pickle
import traceback

import numpy as np
import glom
import tqdm
import setproctitle
import deepmerge
import zmq
import port_for
import ray.services

import lamarckian
from lamarckian.ec import Async
from . import mutation, wrap

NAME = os.path.basename(os.path.dirname(__file__))


class PBT(Async):
    def __init__(self, state, **kwargs):
        super().__init__(state, **kwargs)
        self.ready = list(range(glom.glom(kwargs['config'], 'ec.population')))

    def exploit(self, individual):
        return individual

    def get_variation_param(self, individual, digest):
        return ([individual],), dict(cost=self.cost, operators=[] if individual['digest'] == digest else ['mutation'])

    def new_task(self):
        j = self.ready.pop()
        individual = {**self.population[j], **dict(pbt=j)}
        digest = individual['digest']
        individual = self.exploit(individual)
        individual['pbt'] = j
        args, kwargs = self.get_variation_param(individual, digest)
        self.rpc_async.send('variation', *args, **kwargs)
        return j

    def apply(self, i, child):
        self.population[i] = child

    def __call__(self):
        individual, = self.rpc_async.receive()
        i = individual.pop('pbt')
        assert i not in self.ready
        self.ready.insert(0, i)
        self.apply(i, individual)
        j = self.new_task()
        setproctitle.setproctitle(f"PBT/recv{i}send{j}")
        self.cost += sum(individual['cost'].values())
        return dict(offspring=[individual])


class PBT_(PBT):
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
