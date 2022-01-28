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

import numpy as np
import glom

import lamarckian


def rollout(evaluator):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    def select(population, **kwargs):
        key = glom.glom(kwargs['config'], 'record.rollout.elite', default='result.fitness')
        return max(population, key=lambda individual: glom.glom(individual, key))

    class Evaluator(evaluator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            setattr(self, PATH_FUNC, None)
            self.recorder.register(
                lamarckian.rl.record.Rollout.counter(**kwargs),
                lambda *_, **__: lamarckian.rl.record.Rollout.new(**kwargs)(self.cost, select(self.population, **kwargs)['decision']['blob']),
            )
    return Evaluator
