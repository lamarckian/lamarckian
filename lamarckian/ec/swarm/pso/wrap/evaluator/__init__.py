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
import inspect
import types
import collections.abc

import numpy as np
import glom

CODING = os.path.basename(os.path.dirname(__file__))


def blob(coding=CODING):
    NAME_FUNC = f'{inspect.getframeinfo(inspect.currentframe()).function}.{coding}'
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    def decorate(evaluator):
        class Evaluator(evaluator):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, PATH_FUNC)
                config = kwargs['config']['pso'][coding]
                min = config['velocity'].get('min', 0)
                setattr(self, PATH_FUNC, types.SimpleNamespace(
                    velocity=types.SimpleNamespace(min=min, range=config['velocity']['max'] - min),
                    weight=config['weight'],
                    rs=np.random.RandomState(glom.glom(kwargs['config'], 'seed', default=None)),
                ))

            def initialization(self):
                attr = getattr(self, PATH_FUNC)
                population = super().initialization()
                for individual in population:
                    individual['pso'] = individual.get('pso', dict(velocity={}))
                    individual['pso']['velocity'][coding] = [attr.velocity.min + attr.rs.random_sample(decision.shape) * attr.velocity.range for decision in individual['decision'][coding]]
                return population

            def variation(self, ancestor):
                attr = getattr(self, PATH_FUNC)
                for individual in ancestor:
                    decision = individual['decision'][coding]
                    velocity = individual['pso']['velocity'][coding] = [
                        attr.weight['inertia'] * velocity +
                        attr.weight['pbest'] * attr.rs.uniform(size=decision.shape) * (pbest - decision) +
                        attr.weight['gbest'] * attr.rs.uniform(size=decision.shape) * (gbest - decision)
                        for decision, velocity, pbest, gbest in zip(decision, individual['pso']['velocity'][coding], individual['pso']['pbest'][coding], individual['pso']['gbest'][coding])]
                    individual['decision'][coding] = [decision + velocity for decision, velocity in zip(decision, velocity)]
                offspring = super().variation(ancestor)
                for parent, child in zip(ancestor, offspring):
                    child['pso'] = parent['pso']
                return offspring
        return Evaluator
    return decorate


def real(coding=CODING):
    NAME_FUNC = f'{inspect.getframeinfo(inspect.currentframe()).function}.{coding}'
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    def decorate(evaluator):
        class Evaluator(evaluator):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, PATH_FUNC)
                config = kwargs['config']['pso'][coding]
                min = config['velocity'].get('min', 0)
                encoding = self.describe()[coding]
                if isinstance(encoding, collections.abc.Mapping):
                    lower, upper = np.stack(list(encoding.values())).T
                elif isinstance(encoding, np.ndarray):
                    lower, upper = encoding.T
                else:
                    raise TypeError(type(encoding))
                setattr(self, PATH_FUNC, types.SimpleNamespace(
                    velocity=types.SimpleNamespace(min=min, range=config['velocity']['max'] - min),
                    weight=config['weight'],
                    lower=lower, upper=upper,
                    rs=np.random.RandomState(glom.glom(kwargs['config'], 'seed', default=None)),
                ))

            def initialization(self):
                attr = getattr(self, PATH_FUNC)
                population = super().initialization()
                for individual in population:
                    individual['pso'] = individual.get('pso', dict(velocity={}))
                    individual['pso']['velocity'][coding] = attr.velocity.min + attr.rs.random_sample(individual['decision'][coding].shape) * attr.velocity.range
                return population

            def variation(self, ancestor):
                attr = getattr(self, PATH_FUNC)
                for individual in ancestor:
                    decision = individual['decision'][coding]
                    velocity = individual['pso']['velocity'][coding] = \
                        attr.weight['inertia'] * individual['pso']['velocity'][coding] + \
                        attr.weight['pbest'] * attr.rs.uniform(size=decision.shape) * (individual['pso']['pbest'][coding] - decision) + \
                        attr.weight['gbest'] * attr.rs.uniform(size=decision.shape) * (individual['pso']['gbest'][coding] - decision)
                    individual['decision'][coding] = np.clip(decision + velocity, a_min=attr.lower, a_max=attr.upper)
                offspring = super().variation(ancestor)
                for parent, child in zip(ancestor, offspring):
                    child['pso'] = parent['pso']
                return offspring
        return Evaluator
    return decorate
