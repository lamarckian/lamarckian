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
import collections
import itertools
import enum

import numpy as np

from . import trajectory, reward, objective


def unary(size=4):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    def decorate(mdp):
        class MDP(mdp):
            State = enum.Enum('State', zip('me enemy'.split() + sum([f'x{i} y{i}'.split() for i in reversed(range(size))], []), itertools.count()))

            class Controller(mdp.Controller):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    assert not hasattr(self, PATH_FUNC)
                    setattr(self, PATH_FUNC, collections.deque([self.get_position_ball()] * size, maxlen=size))

                def get_state(self):
                    attr = getattr(self, PATH_FUNC)
                    position = self.get_position_ball()
                    if self.me:
                        position[0] = 1 - position[0]
                    attr.append(position)
                    return dict(inputs=[np.concatenate([[self.get_position_bat(), self.get_position_bat_enemy()], np.concatenate(attr)])])

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.env._get_screen_img_double_player = lambda: [None, None]

            def describe_blob(self):
                encoding = super().describe_blob()
                header = list(self.State.__members__.keys())
                for model in encoding['models']:
                    model['inputs'] = [dict(
                        shape=(len(header),),
                        header=[header],
                    )]
                return encoding
        return MDP
    return decorate
