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
import logging


def fire(mdp):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    def wrap(env):
        def decorate(func):
            def _(action, *args, **kwargs):
                ram = env.unwrapped._get_ram()
                if ram[54] == 0 and ram[11] <= 3:
                    action = env.unwrapped.get_action_meanings().index('FIRE')
                return func(action, *args, **kwargs)
            return _
        return decorate

    class MDP(mdp):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            setattr(self, PATH_FUNC, None)
            self.env.step = wrap(self.env)(self.env.step)
    return MDP


def easter_egg(mdp):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    def wrap(env):
        def decorate(func):
            def _(*args, **kwargs):
                state, reward, done, info = func(*args, **kwargs)
                if len(state.shape) == 3:
                    image = state
                else:
                    image = env.unwrapped._get_image()
                if image[29, 0, 0] != 236:
                    logging.warning(f'pong easter egg detected')
                    done = True
                return state, reward, done, info
            return _
        return decorate

    class MDP(mdp):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            setattr(self, PATH_FUNC, None)
            self.env.step = wrap(self.env)(self.env.step)
    return MDP
