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
import enum
import itertools


def action(mdp):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    NAME_TRAJ = 'trajectory_' + NAME_FUNC

    class MDP(mdp):
        class Controller(mdp.Controller):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, NAME_TRAJ)
                setattr(self, NAME_TRAJ, [])
                self.Action = enum.Enum('Action', zip(self.mdp.describe_blob()['models'][0]['action_name'], itertools.count()))

            async def __call__(self, *args, **kwargs):
                action, = map(int, kwargs['discrete'])
                getattr(self, NAME_TRAJ).append(self.Action(action))
                return await super().__call__(*args, **kwargs)
    return MDP
