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

from ... import Action


def action(mdp):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    NAME_TRAJ = 'trajectory_' + NAME_FUNC

    class MDP(mdp):
        class Controller(mdp.Controller):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, NAME_TRAJ)
                setattr(self, NAME_TRAJ, [])

            async def __call__(self, *args, **kwargs):
                action, = map(int, kwargs['discrete'])
                getattr(self, NAME_TRAJ).append(Action(action))
                return await super().__call__(*args, **kwargs)
    return MDP


def bat(mdp):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    NAME_TRAJ = 'trajectory_' + NAME_FUNC

    class MDP(mdp):
        class Controller(mdp.Controller):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, NAME_TRAJ)
                setattr(self, NAME_TRAJ, [self.get_position_bat()])

            async def __call__(self, *args, **kwargs):
                exp = await super().__call__(*args, **kwargs)
                getattr(self, NAME_TRAJ).append(self.get_position_bat())
                return exp
    return MDP
