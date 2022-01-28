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


def observation(mdp):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    
    class MDP(mdp):
        class Controller(mdp.Controller):
            def get_state(self):
                state = super().get_state()
                assert NAME_FUNC not in state
                state[NAME_FUNC] = self.mdp.env.unwrapped.observation()[self.me]
                return state
    return MDP
