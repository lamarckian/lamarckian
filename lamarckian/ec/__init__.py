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
import itertools
import collections.abc
import types
import pickle
import hashlib
import logging

import numpy as np
import tqdm
import glom
import ray

import lamarckian
from . import crossover, mutation, mating, selection, util, wrap, record


def get_parallel(parallel, **kwargs):
    try:
        return glom.glom(kwargs['config'], 'evaluator.parallel')
    except KeyError:
        return int(ray.cluster_resources()['CPU'] / glom.glom(kwargs['config'], 'evaluator.ray.num_cpus', default=1)) // parallel


class EC(lamarckian.evaluator.Evaluator):
    def initialization(self, size):
        raise NotImplementedError()

    def evaluation(self, population, **kwargs):
        raise NotImplementedError()

    def evolving(self):
        return types.SimpleNamespace(close=lambda: False)

    def __setstate__(self, state):
        self.cost = state.get('cost', 0)
        return state

    def __getstate__(self):
        return dict(cost=self.cost, encoding=self.describe())

    def ensure_encoding(self, population):
        encoding = self.describe()
        _population = self.initialization(len(population))
        for individual, _individual in zip(population, _population):
            decision, _decision = individual['decision'], _individual['decision']
            for coding, value in _decision.items():
                if coding in decision:
                    if isinstance(decision[coding], np.ndarray):
                        if decision[coding].shape != _decision[coding].shape:
                            decision[coding] = value
                        if isinstance(encoding[coding], collections.abc.Mapping):
                            lower, upper = np.stack(list(encoding[coding].values())).T
                        elif isinstance(encoding[coding], np.ndarray):
                            lower, upper = encoding[coding].T
                        else:
                            raise TypeError(type(encoding))
                        decision[coding] = np.clip(decision[coding], lower, upper)
                else:
                    decision[coding] = value
            for coding in [coding for coding in decision if coding not in _decision]:
                del decision[coding]
            individual['digest'] = hashlib.md5(pickle.dumps(individual['decision'])).hexdigest()
        return population


@lamarckian.rl.wrap.remote.parallel()
@lamarckian.util.rpc.wrap.map_methods.all('set')
@lamarckian.util.rpc.wrap.map_methods.any('describe', 'initialize', 'get')
class Sync(EC):
    def __init__(self, state, **kwargs):
        super().__init__(state, **kwargs)
        self.evaluators = self.create_evaluator(**kwargs)
        self.rpc_all = lamarckian.util.rpc.All(self.evaluators, **kwargs)
        self.rpc_any = lamarckian.util.rpc.Any(self.evaluators, **kwargs)
        if state:
            self.population = self.regularize_population(state)
        else:
            self.population = self.initialization(glom.glom(kwargs['config'], 'ec.population'))

    def create_evaluator(self, **kwargs):
        try:
            parallel = glom.glom(kwargs['config'], 'ec.parallel')
        except KeyError:
            parallel = min(int(ray.cluster_resources()['CPU']), glom.glom(kwargs['config'], 'ec.population'))
        assert parallel > 0, parallel
        specs = list(map(lamarckian.util.parse.instance, glom.glom(kwargs['config'], 'evaluator.create')))
        specs.insert(1, lamarckian.ec.wrap.evaluator.variator)
        Evaluator = lamarckian.evaluator.parse(*specs, **kwargs)
        Evaluator = lamarckian.util.rpc.wrap.all(Evaluator)
        Evaluator = lamarckian.util.rpc.wrap.any(Evaluator)
        root = kwargs.get('root', None)
        record = glom.glom(kwargs['config'], 'ec.train.record')
        cls = Evaluator.remote_cls(**kwargs)

        def create(index, parallel):
            try:
                name = f"{kwargs['name']}/evaluator{index}"
            except KeyError:
                name = f"evaluator{index}"
            return cls.options(name=name).remote(**{**kwargs, **dict(
                index=index, parallel=parallel, name=name,
                root=os.path.join(root, 'evaluator', str(index)) if root is not None and index < record else None,
            )})
        return [create(index, get_parallel(parallel, **kwargs)) for index in range(parallel)]

    def initialization(self, size):
        population = list(itertools.chain(*self.rpc_any.map((('initialization', (), {}) for _ in range(size)))))
        digests = set(individual['digest'] for individual in population)
        if len(digests) < size:
            logging.warning(f"unique individuals ({len(digests)}) are less than population size ({size})")
        return population
        population = {}
        while len(population) < size:
            for individual in itertools.chain(*self.rpc_any.map((('initialization', (), {}) for _ in range(size - len(population))))):
                population[individual['digest']] = individual
        return list(population.values())[:size]

    def evaluation(self, population, **kwargs):
        tasks = [('evaluation', ([individual],), {}) for individual in population]
        if kwargs:
            tasks = tqdm.tqdm(tasks, **kwargs)
        population = list(itertools.chain(*self.rpc_any.map(iter(tasks))))
        self.cost += sum(sum(individual['cost'].values()) for individual in population)
        return population

    def evaluation0(self, *args, **kwargs):
        return self.evaluation(*args, **kwargs)

    def evolving(self):
        population = sum(map(util.population.transfer, glom.glom(self.kwargs, 'ec.transfer', default=[])), [])
        if population:
            logging.warning(f'transfer {len(population)} individuals into the population')
        self.population = population + self.population
        size = glom.glom(self.kwargs['config'], 'ec.population')
        if len(self.population) < size:
            self.population += self.initialization(size - len(self.population))
        self.population = self.population[:size]
        self.population = self.ensure_encoding(self.population)
        self.population = self.evaluation0(self.population, desc='evaluation')
        return super().evolving()

    def regularize_population(self, state):
        population = state['population']
        population0 = self.initialization(len(population))
        for coding, encoding in self.describe().items():
            if coding in {'real', 'integer'} and isinstance(encoding, dict):
                reset = set(glom.glom(self.kwargs['config'], f"ec.encoding.{coding}.reset", default=[]))
                for i, (individual, individual0) in enumerate(zip(population, population0)):
                    if coding in state['encoding']:
                        hparam = {key: individual['decision'][coding][i] for i, key in enumerate(state['encoding'][coding])}
                        decision = []
                        for (key, boundary), value in zip(encoding.items(), individual0['decision'][coding]):
                            if key in reset:
                                decision.append(value)
                            else:
                                decision.append(np.clip(hparam.get(key, value), *boundary))
                        assert len(decision) == len(encoding), (len(decision), len(encoding))
                        individual['decision'][coding] = np.array(decision)
                    else:
                        individual['decision'][coding] = individual0['decision'][coding]
        return population

    def __setstate__(self, state):
        self.population = self.regularize_population(state)
        return super().__setstate__(state)

    def __getstate__(self):
        state = super().__getstate__()
        state['population'] = self.population
        return state


class Async(Sync):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rpc_async = lamarckian.util.rpc.wrap.any_count(lamarckian.util.rpc.Any)(self.evaluators, **kwargs)

    def new_task(self):
        return self.rpc_async.send('variation', self.mating(), cost=self.cost)

    def evolving(self):
        evolving = super().evolving()
        for _ in range(len(self.evaluators)):
            self.new_task()

        def close():
            for _ in range(len(self.rpc_async)):
                self.rpc_async.receive()
            return evolving.close()
        return types.SimpleNamespace(close=close)

    def __next__(self):
        offspring = self.rpc_async.receive()
        self.new_task()
        self.cost += sum(sum(individual['cost'].values()) for individual in offspring)
        return [individual for individual in offspring if 'result' in individual]

    def breeding(self, size):
        offspring = []
        while len(offspring) < size:
            offspring += self.__next__()
        return offspring

    def mating(self):
        raise NotImplementedError()


from . import ea
