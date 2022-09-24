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

import os
import inspect
import types
import collections

import numpy as np
import glom

import lamarckian
CODING = os.path.basename(os.path.dirname(__file__))


def blob(coding=CODING):
    NAME_FUNC = f'{inspect.getframeinfo(inspect.currentframe()).function}.{coding}'
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    def decorate(evaluator):
        class Evaluator(evaluator):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, PATH_FUNC)
                config = kwargs['config']['ec']['swarm']['cso'][coding]
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
                    individual['cso'] = individual.get('cso', dict(velocity={}))
                    individual['cso']['velocity'][coding] = [attr.velocity.min + attr.rs.random_sample(decision.shape) * attr.velocity.range for decision in individual['decision'][coding]]
                return population

            def variation(self, ancestor):
                attr = getattr(self, PATH_FUNC)
                loser, = ancestor
                decision = loser['decision'][coding]
                velocity = loser['cso']['velocity'][coding] = [
                    attr.weight['inertia'] * attr.rs.uniform(size=decision.shape) * velocity +
                    attr.weight['winner'] * attr.rs.uniform(size=decision.shape) * (winner - decision) +
                    attr.weight['center'] * attr.rs.uniform(size=decision.shape) * (center - decision)
                    for decision, velocity, winner, center in zip(decision, loser['cso']['velocity'][coding], loser['cso']['winner'][coding], loser['cso']['center'][coding])]
                loser['decision'][coding] = [decision + velocity for decision, velocity in zip(decision, velocity)]
                return super().variation([loser])
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
                config = kwargs['config']['ec']['swarm']['cso'][coding]
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
                    individual['cso'] = individual.get('cso', dict(velocity={}))
                    individual['cso']['velocity'][coding] = attr.velocity.min
                    # individual['cso']['velocity'][coding] = attr.velocity.min + attr.rs.random_sample(individual['decision'][coding].shape) * attr.velocity.range
                return population

            def variation(self, ancestor):
                attr = getattr(self, PATH_FUNC)
                loser, = ancestor
                decision = loser['decision'][coding]
                velocity = loser['cso']['velocity'][coding] = \
                    attr.weight['inertia'] * attr.rs.uniform(size=decision.shape) * loser['cso']['velocity'][coding] + \
                    attr.weight['winner'] * attr.rs.uniform(size=decision.shape) * (loser['cso']['winner'][coding] - decision) + \
                    attr.weight['center'] * attr.rs.uniform(size=decision.shape) * (loser['cso']['center'][coding] - decision)
                loser['decision'][coding] = np.clip(decision + velocity, a_min=attr.lower, a_max=attr.upper)
                return super().variation([loser])
        return Evaluator
    return decorate
