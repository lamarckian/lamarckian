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

import os
import contextlib
import itertools

import torch

import lamarckian
import lamarckian.mdp.gfootball as module

ROOT = os.path.dirname(__file__)


class Agent(object):
    def __init__(self, env, me, actions):
        self.env = env
        self.me = me
        self.actions = actions

    def __call__(self, state):
        return dict(action=torch.randint(0, self.actions - 1, [1]))


def test_pve():
    sample = 2
    config = {}
    for path in [
        os.path.dirname(lamarckian.__file__) + '.yml',
        os.path.join(ROOT, 'simple115.yml'),
    ]:
        config = lamarckian.util.config.read(path, config)
    with contextlib.closing(module.PvE(config=config)) as mdp:
        for me in range(len(mdp)):
            encoding = mdp.describe()
            agent = Agent(mdp, me, encoding['blob']['models'][me]['kwargs']['outputs']['discrete'])
            torch.manual_seed(0)
            _, results = lamarckian.mdp.evaluate(mdp, me, agent, itertools.repeat({}, sample))
            result = mdp.reduce(results)
            torch.manual_seed(0)
            _, _result = lamarckian.mdp.evaluate(mdp, me, agent, itertools.repeat({}, sample))
            assert result['fitness'] == _result['fitness'], (result, _result)
