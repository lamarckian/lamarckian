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
import types
import logging

import glom

import lamarckian


def set_opponents_train(ec):
    class EC(ec):
        def evaluation0(self, population, *args, **kwargs):
            enemy, = glom.glom(self.kwargs['config'], 'rl.enemies', default=[1])
            self.rpc_all('set_opponents_train', [{enemy: individual['decision']['blob']} for individual in population])
            return super().evaluation0(population, *args, **kwargs)

        def __call__(self, *args, **kwargs):
            outcome = super().__call__(*args, **kwargs)
            enemy, = glom.glom(self.kwargs['config'], 'rl.enemies', default=[1])
            self.rpc_all('set_opponents_train', [{enemy: individual['decision']['blob']} for individual in self.population])
            return outcome
    return EC


def strict(stopper='rl.opponent.eval.stopper'):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    def reevaluate(self, population):
        variation = []
        evaluation = []
        for individual in population:
            offspring = self.rpc_async.receive()
            self.rpc_async.send('evaluation', [{**individual, **{PATH_FUNC: None}}])
            if len(offspring) == 1 and PATH_FUNC in offspring[0]:
                evaluation += offspring
            else:
                variation.append(offspring)
        while len(evaluation) < len(population):
            offspring = self.rpc_async.receive()
            if len(offspring) == 1 and PATH_FUNC in offspring[0]:
                evaluation += offspring
            else:
                variation.append(offspring)
        for offspring in variation:
            self.rpc_async.send(None, offspring)
        return evaluation

    def decorate(ec):
        class EC(ec):
            def evaluation0(self, population, *args, **kwargs):
                enemy, = glom.glom(self.kwargs['config'], 'rl.enemies', default=[1])
                opponents = [{enemy: individual['decision']['blob']} for individual in population]
                self.rpc_all('set_opponents_train', opponents)
                digest, = set(self.rpc_all.fetch_all('set_opponents_eval', opponents))
                assert digest, digest
                logging.info(f'initial opponents_eval={digest}')
                cls = lamarckian.evaluator.parse(*glom.glom(self.kwargs['config'], stopper), **self.kwargs)
                make_stopper = lambda: cls(self, **self.kwargs)
                attr = types.SimpleNamespace(
                    digest_opponents_eval=digest,
                    make_stopper=make_stopper,
                    stopper=make_stopper(),
                )
                setattr(self, PATH_FUNC, attr)
                return super().evaluation0(population, *args, **kwargs)

            def __next__(self):
                attr = getattr(self, PATH_FUNC)
                offspring = self.rpc_async.receive()
                sum(sum(individual.get('cost', {}).values()) for individual in offspring)
                self.cost += sum(sum(individual['cost'].values()) for individual in offspring)
                offspring = [individual for individual in offspring if 'result' in individual]
                digest, = set([individual['result']['digest_opponents_eval'] for individual in offspring])
                if digest == attr.digest_opponents_eval:
                    self.new_task()
                    return offspring
                else:
                    logging.warning(f'reevaluate offspring because digest_opponents_eval ({digest}) != {attr.digest_opponents_eval}')
                    self.rpc_async.send('evaluation', offspring)
                    return []

            def __call__(self, *args, **kwargs):
                attr = getattr(self, PATH_FUNC)
                digest, = set([individual['result']['digest_opponents_eval'] for individual in self.population])
                if digest != attr.digest_opponents_eval:
                    logging.warning(f'reevaluate population with opponents_eval={attr.digest_opponents_eval}')
                    self.population = reevaluate(self, self.population)
                    digest, = set([individual['result']['digest_opponents_eval'] for individual in self.population])
                    assert digest == attr.digest_opponents_eval, (digest, attr.digest_opponents_eval)
                outcome = super().__call__(*args, **kwargs)
                enemy, = glom.glom(self.kwargs['config'], 'rl.enemies', default=[1])
                opponents = [{enemy: individual['decision']['blob']} for individual in self.population]
                self.rpc_all('set_opponents_train', opponents)
                if attr.stopper(outcome):
                    attr.stopper = attr.make_stopper()
                    attr.digest_opponents_eval, = set(self.rpc_all.fetch_all('set_opponents_eval', opponents))
                    assert attr.digest_opponents_eval, attr.digest_opponents_eval
                return outcome
        return EC
    return decorate
