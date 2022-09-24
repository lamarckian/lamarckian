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
import types
import copy
import random
import logging

import numpy as np
import glom

import lamarckian


def global_best(swarm):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    class Swarm(swarm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            setattr(self, PATH_FUNC, random.Random(glom.glom(kwargs['config'], 'seed', default=None)))
            self.recorder.register(lamarckian.util.counter.Time(**glom.glom(kwargs['config'], 'record.scalar')), lambda: lamarckian.util.record.Scalar(self.cost, **{'pso/gbest': getattr(self, PATH_FUNC).gbest['fitness']}))
            self.recorder.register(lamarckian.util.counter.Time(**kwargs['config']['record']['histogram']), lambda: lamarckian.util.record.Histogram(self.cost, **{'pso/pbest': np.array([pbest['fitness'] for pbest in getattr(self, PATH_FUNC).pbest])}))

        def evaluation0(self, *args, **kwargs):
            population = super().evaluation0(*args, **kwargs)
            keys = set(population[0]['pso']['velocity'])
            swarm = [dict(decision={key: value for key, value in individual['decision'].items() if key in keys}, fitness=individual['result']['fitness']) for individual in population]
            setattr(self, PATH_FUNC, types.SimpleNamespace(
                keys=keys,
                velocity=[individual['pso']['velocity'] for individual in population],
                swarm=swarm,
                pbest=copy.copy(swarm),
                gbest=swarm[np.argmax([particle['fitness'] for particle in swarm])],
            ))
            return population

        def put(self, evaluator, offspring):
            attr = getattr(self, PATH_FUNC)
            ancestor = [{**self.population[index], **dict(pso=dict(
                index=index,
                velocity=attr.velocity[index], pbest=attr.pbest[index]['decision'],
                gbest=attr.gbest['decision'],
            ))} for index in getattr(self, PATH_FUNC).sample(range(len(self.population)), self.choose)]
            for parent in ancestor:
                particle = attr.swarm[parent['pso']['index']]['decision']
                parent['decision'] = {key: particle[key] if key in attr.keys else value for key, value in parent['decision'].items()}
            return evaluator.variation.remote(ancestor)

        def __next__(self):
            attr = getattr(self, PATH_FUNC)
            offspring = super().__next__()
            for child in offspring:
                pso = child['pso']
                index = pso['index']
                attr.velocity[index] = pso['velocity']
                attr.swarm[index] = particle = dict(decision={key: value for key, value in child['decision'].items() if key in attr.keys}, fitness=child['result']['fitness'])
                if particle['fitness'] > attr.pbest[index]['fitness']:
                    attr.pbest[index] = particle
                if particle['fitness'] > attr.gbest['fitness']:
                    attr.gbest = particle
                    logging.info(f'gbest={particle["fitness"]}')
            return offspring
    return Swarm
