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

import numpy as np
import glom


def lazy(mdp):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    KEY = f'reward.{NAME_FUNC}'

    class MDP(mdp):
        class Controller(mdp.Controller):
            def get_reward(self):
                return super().get_reward() + self.mdp.hparam[KEY] * (0 if self.trajectory_action[-1] in {self.Action.NOOP, self.Action.FIRE} else -1)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.hparam.setup(KEY, glom.glom(kwargs['config'], f'mdp.gym.breakout.reward.{NAME_FUNC}', default=1), np.float)
    return MDP


def busy(mdp):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    KEY = f'reward.{NAME_FUNC}'

    class MDP(mdp):
        class Controller(mdp.Controller):
            def get_reward(self):
                return super().get_reward() + self.mdp.hparam[KEY] * (1 if self.trajectory_action[-1] in {self.Action.RIGHT, self.Action.LEFT} else 0)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.hparam.setup(KEY, glom.glom(kwargs['config'], f'mdp.gym.breakout.reward.{NAME_FUNC}', default=1), np.float)
    return MDP
