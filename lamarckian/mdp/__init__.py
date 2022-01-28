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

import types
import contextlib
import asyncio

import numpy as np

from . import rollout, util, wrap


class MDP(object):
    class Controller(object):
        def __init__(self, mdp):
            self.mdp = mdp

        def get_state(self):
            raise NotImplementedError()

        async def __call__(self, exp):
            raise NotImplementedError()

        def get_reward(self):
            raise NotImplementedError()

        def get_result(self):
            raise NotImplementedError()

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def close(self):
        pass

    def describe(self):
        raise NotImplementedError()

    def initialize(self):
        raise NotImplementedError()

    def set(self, decision):
        raise NotImplementedError()

    def get(self):
        raise NotImplementedError()

    def seed(self, seed):
        pass

    def evaluating(self, seed):
        return types.SimpleNamespace(close=lambda: None)

    def __len__(self):
        raise NotImplementedError()

    def reset(self, *args, **kwargs):
        raise NotImplementedError()

    def __getstate__(self):
        return dict(encoding=self.describe())

    def __setstate__(self, state):
        pass


def evaluate(mdp, me, agent, opponents, loop=asyncio.get_event_loop()):
    cost = 0
    results = []
    for seed, opponent in enumerate(opponents):
        with contextlib.closing(mdp.evaluating(seed)):
            battle = mdp.reset(me, *opponent, loop=loop)
            with contextlib.closing(battle):
                costs = loop.run_until_complete(asyncio.gather(
                    rollout.get_cost(battle.controllers[0], agent),
                    *[rollout.get_cost(controller, agent) for controller, agent in zip(battle.controllers[1:], opponent.values())],
                    *battle.ticks,
                ))[:len(battle.controllers)]
                cost += max(costs)
                results.append(battle.controllers[0].get_result())
    return cost, results
