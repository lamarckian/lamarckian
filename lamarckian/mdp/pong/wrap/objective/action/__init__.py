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
import types

import numpy as np
import glom

from .... import Action


def lazy(mdp):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    class MDP(mdp):
        class Controller(mdp.Controller):
            def get_result(self):
                result = super().get_result()
                result[NAME_FUNC] = sum(1 for action in self.trajectory_action if action == Action.stop) / len(self.trajectory_action)
                result['objective'].append(np.array([result['fitness'], result[NAME_FUNC]]) @ getattr(self.mdp, PATH_FUNC))
                return result

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            setattr(self, PATH_FUNC, np.array(glom.glom(kwargs['config'], f"mdp.pong.objective.{NAME_FUNC}.weight")))
    return MDP


def busy(mdp):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    class MDP(mdp):
        class Controller(mdp.Controller):
            def get_result(self):
                result = super().get_result()
                result[NAME_FUNC] = sum(1 for action in self.trajectory_action if action != Action.stop) / len(self.trajectory_action)
                result['objective'].append(np.array([result['fitness'], result[NAME_FUNC]]) @ getattr(self.mdp, PATH_FUNC))
                return result

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            setattr(self, PATH_FUNC, np.array(glom.glom(kwargs['config'], f"mdp.pong.objective.{NAME_FUNC}.weight")))
    return MDP
