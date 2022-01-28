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
import collections
import itertools
import enum
import types

import numpy as np


def get_position(ram):
    return types.SimpleNamespace(
        me=ram[70] / 184,
        x=(ram[99] - 56) / 144,
        y=(ram[101] - 87) / 93,
    )

from . import rule, trajectory, reward, objective


def unary(size=4):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    def decorate(mdp):
        class MDP(mdp):
            State = enum.Enum('State', zip(['me'] + sum([f'x{i} y{i}'.split() for i in reversed(range(size))], []), itertools.count()))

            class Controller(mdp.Controller):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    assert not hasattr(self, PATH_FUNC)
                    position = get_position(self.mdp.env.unwrapped._get_ram())
                    setattr(self, PATH_FUNC, collections.deque([np.array([position.x, position.y])] * size, maxlen=size))

                def get_state(self):
                    attr = getattr(self, PATH_FUNC)
                    position = get_position(self.mdp.env.unwrapped._get_ram())
                    attr.append(np.array([position.x, position.y]))
                    return dict(inputs=[np.concatenate([[position.me], np.concatenate(attr)])])

            def describe_blob(self):
                header = list(self.State.__members__.keys())
                encoding = super().describe_blob()
                for model in encoding['models']:
                    model['inputs'] = [dict(shape=(len(header),), header=header)]
                return encoding
        return MDP
    return decorate
