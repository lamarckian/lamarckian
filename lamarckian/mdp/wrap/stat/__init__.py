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
import torch
import glom

import lamarckian


def trajectory(mdp):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    class MDP(mdp):
        class Controller(mdp.Controller):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, PATH_FUNC)
                setattr(self, PATH_FUNC, 0)

            def get_reward(self):
                setattr(self, PATH_FUNC, getattr(self, PATH_FUNC) + 1)
                return super().get_reward()

            def get_result(self):
                result = super().get_result()
                attr = getattr(self, PATH_FUNC)
                if attr > 0:
                    result[NAME_FUNC] = attr
                return result
    return MDP


def credit(mdp):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    class MDP(mdp):
        class Controller(mdp.Controller):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, PATH_FUNC)
                setattr(self, PATH_FUNC, [])

            def get_reward(self):
                reward = super().get_reward()
                getattr(self, PATH_FUNC).append(reward)
                return reward

            def get_result(self):
                attr = getattr(self.mdp, PATH_FUNC)
                result = super().get_result()
                for name, discount, reward in zip(attr.names, attr.discounts, np.array(getattr(self, PATH_FUNC)).T):
                    result[f"{NAME_FUNC}/{name}"] = lamarckian.rl.cumulate(torch.from_numpy(reward), discount)[0]
                return result

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            encoding = self.describe()['blob']
            discount = glom.glom(kwargs['config'], 'rl.discount')
            setattr(self, PATH_FUNC, types.SimpleNamespace(
                names=encoding['reward'],
                discounts=np.array([glom.glom(kwargs['config'], f"rl.discount_{name}", default=discount) for name in encoding['reward']]),
            ))
    return MDP
