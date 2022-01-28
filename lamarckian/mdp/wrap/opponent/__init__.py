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

import numpy as np
import glom

import lamarckian


def random(section='mdp.opponent.random'):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    def load(**kwargs):
        items = []
        for s in glom.glom(kwargs['config'], section):
            try:
                s, weight = s.rsplit('=', 1)
                weight = float(weight)
            except ValueError:
                weight = 1
            items.append((lamarckian.util.parse.instance(s), weight))
        agents, probs = zip(*items)
        return agents, np.array(probs) / np.sum(probs)

    def decorate(mdp):
        class Problem(mdp):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, PATH_FUNC)
                agents, probs = load(**kwargs)
                setattr(self, PATH_FUNC, types.SimpleNamespace(agents=agents, probs=probs))

            def reset(self, *args, **kwargs):
                attr = getattr(self, PATH_FUNC)
                agent = np.random.choice(attr.agents, p=attr.probs)
                opponents = set(range(len(self))) - set(args)
                assert opponents
                controllers, ticks = super().reset(*args, *opponents, **kwargs)
                return controllers[:len(args)], [agent(self.controllers[me]) for me in opponents] + ticks
        return Problem
    return decorate
