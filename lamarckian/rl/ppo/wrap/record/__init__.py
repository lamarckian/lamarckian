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
import contextlib

import numpy as np
import glom

import lamarckian


def stale(tag):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    def fetch(self):
        attr = getattr(self, PATH_FUNC)
        if attr:
            try:
                return {
                    f'{tag}/{NAME_FUNC}/max': np.max(attr),
                    f'{tag}/{NAME_FUNC}/mean': np.mean(attr),
                }
            finally:
                setattr(self, PATH_FUNC, [])
        else:
            return {}

    def decorate(rl):
        class RL(rl):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, PATH_FUNC)
                setattr(self, PATH_FUNC, [])
                self.recorder.register(
                    lamarckian.util.counter.Time(**glom.glom(kwargs['config'], 'record.scalar')),
                    lambda *args, **kwargs: lamarckian.util.record.Scalar(self.cost, **fetch(self)),
                )

            def receive(self):
                cost, tensors, results, iteration = super().receive()
                getattr(self, PATH_FUNC).append(self.iteration - iteration)
                return cost, tensors, results, iteration
        return RL
    return decorate
