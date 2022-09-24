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
import itertools

import numpy as np
import glom

import lamarckian


def topology(rl):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function

    class RL(rl):
        class Learner(rl.Learner):
            def get_topology(self):
                return repr(self.rpc_all)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.recorder.put(lamarckian.util.record.Text(self.cost, **{f"{NAME_FUNC}{i}": topology for i, topology in enumerate(self.ray.get([learner.get_topology.remote() for learner in self.learners]))}))
    return RL


def stale(tag):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    def fetch(self):
        iterations = getattr(self, PATH_FUNC)
        if iterations:
            try:
                return {
                    f'{tag}/{NAME_FUNC}/max': np.max(iterations),
                    f'{tag}/{NAME_FUNC}/mean': np.mean(iterations),
                }
            finally:
                setattr(self, PATH_FUNC, [])
        else:
            return {}

    def decorate(rl):
        class RL(rl):
            class Learner(rl.Learner):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    assert not hasattr(self, PATH_FUNC)
                    setattr(self, PATH_FUNC, [])

                def __next__(self):
                    cost, tensors, results, iterations = super().__next__()
                    for iteration in iterations:
                        getattr(self, PATH_FUNC).append(self.iteration - iteration)
                    return cost, tensors, results, iterations

                def __call__(self):
                    cost, outcome = super().__call__()
                    outcome[PATH_FUNC] = getattr(self, PATH_FUNC)
                    setattr(self, PATH_FUNC, [])
                    return cost, outcome

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, PATH_FUNC)
                setattr(self, PATH_FUNC, [])
                self.recorder.register(
                    lamarckian.util.counter.Time(**glom.glom(kwargs['config'], 'record.scalar')),
                    lambda *args, **kwargs: lamarckian.util.record.Scalar(self.cost, **fetch(self)),
                )

            def __next__(self):
                costs, outcomes = super().__next__()
                setattr(self, PATH_FUNC, getattr(self, PATH_FUNC) + list(itertools.chain(*[outcome.pop(PATH_FUNC) for outcome in outcomes])))
                return costs, outcomes
        return RL
    return decorate


def broadcaster(tag):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    def get(self):
        outcomes = getattr(self, PATH_FUNC)
        results = [result for result in (outcome.pop(PATH_FUNC, {}) for outcome in outcomes) if result]
        if results:
            return lamarckian.util.reduce(results)
        else:
            return {}

    def decorate(rl):
        class RL(rl):
            class Learner(rl.Learner):
                def __call__(self):
                    cost, outcome = super().__call__()
                    try:
                        outcome[PATH_FUNC] = self.broadcaster.profile
                    except AttributeError:
                        pass
                    return cost, outcome

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, PATH_FUNC)
                self.recorder.register(
                    lamarckian.util.counter.Time(**glom.glom(kwargs['config'], 'record.scalar')),
                    lambda outcome: lamarckian.util.record.Scalar(self.cost, **{
                        **{f'{tag}/broadcast/{key}': value for key, value in get(self).items()},
                    }),
                )

            def __next__(self):
                costs, outcomes = super().__next__()
                setattr(self, PATH_FUNC, outcomes)
                return costs, outcomes
        return RL
    return decorate
