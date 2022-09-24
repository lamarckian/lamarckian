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


def bounce(mdp):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'
    KEY = f'reward.{NAME_FUNC}'

    def wrap(ball, controller):
        def decorate(func):
            def _(axis, speed_delta):
                if axis == 'x':
                    if ball._speed_x < 0:
                        setattr(controller, PATH_FUNC, 1)
                    elif ball._speed_x > 0:
                        setattr(controller, PATH_FUNC, -1)
                return func(axis, speed_delta)
            return _
        return decorate

    class MDP(mdp):
        class Controller(mdp.Controller):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, PATH_FUNC)

            async def __call__(self, *args, **kwargs):
                setattr(self, PATH_FUNC, 0)
                return await super().__call__(*args, **kwargs)

            def get_reward(self):
                reward = self.mdp.hparam[KEY] * getattr(self, PATH_FUNC)
                if self.me:
                    reward = -reward
                return np.append(super().get_reward(), reward)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            setattr(self, PATH_FUNC, self.get_ball()._bounce)
            self.hparam.setup(KEY, glom.glom(kwargs['config'], f"mdp.pong.reward.{NAME_FUNC}", default=1), np.float)

        def describe_blob(self):
            encoding = super().describe_blob()
            encoding['reward'].append(NAME_FUNC)
            return encoding

        def reset(self, *args, **kwargs):
            battle = super().reset(*args, **kwargs)
            ball = self.get_ball()
            ball._bounce = wrap(ball, battle.controllers[0])(getattr(self, PATH_FUNC))
            return battle
    return MDP
