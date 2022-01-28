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

import inspect
import pickle
import hashlib
import contextlib
import types
import copy
import logging

import numpy as np
import torch
import glom
import setproctitle

import lamarckian


def variator(evaluator):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    class Evaluator(evaluator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            encoding = self.describe()
            attr = types.SimpleNamespace(
                crossover=lamarckian.ec.crossover.Crossover(encoding, evaluator=self, **kwargs),
                mutation=lamarckian.ec.mutation.wrap.operator(lamarckian.ec.mutation.Mutation)(encoding, evaluator=self, **kwargs),
                detect=glom.glom(kwargs['config'], f'ec.train.detect', default=False),
            )
            try:
                assert callable(self) and type(self).__call__ is not lamarckian.evaluator.Evaluator.__call__
                Stopper = lamarckian.evaluator.parse(*glom.glom(kwargs['config'], 'ec.train.stopper'), **kwargs)
                attr.stopper = lambda: Stopper(self, **kwargs)
            except (AssertionError, KeyError):
                logging.warning('training disabled')
            attr.operators = {key: getattr(attr, key) for key in glom.glom(kwargs['config'], f'ec.operator', default=['crossover', 'mutation'])}
            setattr(self, PATH_FUNC, attr)

        def close(self):
            attr = getattr(self, PATH_FUNC)
            attr.crossover.close()
            attr.mutation.close()
            return super().close()

        def get_choose(self):
            attr = getattr(self, PATH_FUNC)
            return max(getattr(operator, 'choose', 1) for operator in attr.operators.values())

        def initialization(self):
            decision = super().initialize()
            return [dict(
                decision=decision,
                digest=hashlib.md5(pickle.dumps(decision)).hexdigest(),
                age=0,
            )]

        def evaluation(self, population):
            for individual in population:
                cost = self.cost
                decision = individual['decision']
                self.set(decision)
                with contextlib.closing(lamarckian.util.duration.Measure()) as duration:
                    individual['results'] = self.evaluate()
                individual['result'] = self.reduce(individual['results'])
                individual['digest'] = hashlib.md5(pickle.dumps(decision)).hexdigest()
                individual['cost'] = dict(evaluate=self.cost - cost)
                individual['duration'] = dict(evaluate=duration.get())
                individual['age'] = individual.get('age', 0) + sum(individual['cost'].values())
            return population

        def oracle(self, individual, evaluate=True, title=None):
            if title is None:
                title = self.kwargs['name']
            attr = getattr(self, PATH_FUNC)
            cost = self.cost
            self.set(copy.deepcopy(individual['decision']))
            try:
                setproctitle.setproctitle(f"{title}.train")
                episode = 0
                with contextlib.closing(attr.stopper()) as stopper, contextlib.closing(self.training()), contextlib.closing(lamarckian.util.duration.Measure()) as duration, (torch.autograd.detect_anomaly() if attr.detect else contextlib.closing(types.SimpleNamespace(close=lambda: None))):
                    for iteration in range(np.iinfo(np.int).max):
                        outcome = self()
                        episode += len(outcome['results'])
                        if stopper(outcome):
                            break
                individual['decision'] = decision = copy.deepcopy(stopper.get())
                individual['digest'] = hashlib.md5(pickle.dumps(decision)).hexdigest()
                individual['cost'], cost = dict(train=self.cost - cost), self.cost
                individual['duration'] = dict(train=duration.get())
                individual['iteration'] = iteration
                individual['episode'] = episode
                if evaluate:
                    setproctitle.setproctitle(f"{title}.evaluate")
                    with contextlib.closing(lamarckian.util.duration.Measure()) as duration:
                        individual['results'] = stopper.evaluate()
                    individual['result'] = self.reduce(individual['results'])
                    individual['cost']['evaluate'], cost = self.cost - cost, self.cost
                    individual['duration']['evaluate'] = duration.get()
                individual['age'] += sum(individual['cost'].values())
                return individual
            # except:
            #     traceback.print_exc()
            #     return dict(cost=individual['cost'], duration=individual['duration'], exception=traceback.format_exc())
            finally:
                setproctitle.setproctitle(title)

        def variation(self, ancestor, call={}, operators=None, evaluate=True, title=None, cost=None):
            if cost is not None:
                self.cost = cost
            for name, (args, kwargs) in call.items():
                func = getattr(self, name)
                func(*args, **kwargs)
            attr = getattr(self, PATH_FUNC)
            origin = {individual['digest']: individual['result'] for individual in ancestor}
            age = max(individual['age'] for individual in ancestor)
            offspring = ancestor
            for operator in attr.operators.values() if operators is None else [attr.operators[key] for key in operators]:
                offspring = operator(offspring)
            for individual in offspring:
                individual['choose'] = len(ancestor)
                individual['origin'] = origin
                individual['age'] = age
            if hasattr(attr, 'stopper'):
                offspring = [self.oracle(individual, evaluate, title) for individual in offspring]
            elif evaluate:
                offspring = self.evaluation(offspring)
            return offspring
    return Evaluator
