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

import os
import inspect
import types
import datetime

import numpy as np
import glom

from . import blob


def zero(mdp):
    class MDP(mdp):
        class Controller(mdp.Controller):
            def get_reward(self):
                return np.array([0])

        def describe(self):
            encoding = super().describe()
            encoding['blob']['reward'] = ['_']
            return encoding
    return MDP


def agg(mdp):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    class MDP(mdp):
        class Controller(mdp.Controller):
            def get_reward(self):
                attr = getattr(self.mdp, PATH_FUNC)
                reward = super().get_reward()
                assert reward.shape[-1] == len(attr.reward), (reward.shape, attr.reward)
                return np.stack([sum(np.take(reward, i, -1) for i in items.values()) for items in attr.agg.values()], -1)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            reward = super().describe()['blob']['reward']
            assert len(reward) == len(set(reward)), reward
            index = {name: i for i, name in enumerate(reward)}
            agg = {key: {name: index[name] for name in names} for key, names in glom.glom(kwargs['config'], 'mdp.reward.agg', default={}).items() if names}
            if agg:
                remaining = set(index) - {name for names in agg.values() for name in names}
                assert not remaining, ' '.join(remaining)
            else:
                agg = {'': index}
            setattr(self, PATH_FUNC, types.SimpleNamespace(agg=agg, reward=reward))

        def describe(self):
            encoding = super().describe()
            encoding['blob']['reward'] = list(getattr(self, PATH_FUNC).agg)
            return encoding
    return MDP


def log(*names):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    def decorate(mdp):
        class MDP(mdp):
            class Controller(mdp.Controller):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    if hasattr(self.mdp, PATH_FUNC):
                        for name in names:
                            name = f"{NAME_FUNC}.{name}"
                            assert not hasattr(self, name)
                            setattr(self, name, [])

                def get_reward(self):
                    reward = super().get_reward()
                    if hasattr(self.mdp, PATH_FUNC):
                        assert reward.shape[-1] >= len(names), (reward.shape, names)
                        for i, name in enumerate(names):
                            getattr(self, f"{NAME_FUNC}.{name}").append(np.take(reward, -i - 1, -1))
                    return reward

                def get_result(self):
                    if hasattr(self.mdp, PATH_FUNC):
                        root = getattr(self.mdp, PATH_FUNC)
                        os.makedirs(root, exist_ok=True)
                        for name in names:
                            np.savetxt(os.path.join(root, f"{name}.tsv"), getattr(self, f"{NAME_FUNC}.{name}"), fmt='%s', delimiter='\t')
                    return super().get_result()

            def reset(self, *args, **kwargs):
                if glom.glom(self.kwargs['config'], f"mdp.reward.{NAME_FUNC}", default=False):
                    root = os.path.expanduser(os.path.expandvars(glom.glom(self.kwargs['config'], 'root')))
                    setattr(self, PATH_FUNC, os.path.join(root, 'reward', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')))
                return super().reset(*args, **kwargs)
        return MDP
    return decorate


def pbrs(mdp):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    def shaping(i, potential, reward, done, discount):
        if done:
            value = 0
        else:
            value = potential[i] + reward
        reward = discount * value - potential[i]
        potential[i] = value
        return reward

    class MDP(mdp):
        class Controller(mdp.Controller):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, PATH_FUNC)
                discounts = getattr(self.mdp, PATH_FUNC)
                setattr(self, PATH_FUNC, types.SimpleNamespace(potential={i: 0 for i in discounts}))

            async def __call__(self, *args, **kwargs):
                exp = await super().__call__(*args, **kwargs)
                getattr(self, PATH_FUNC).done = exp['done']
                return exp

            def get_reward(self):
                discounts = getattr(self.mdp, PATH_FUNC)
                attr = getattr(self, PATH_FUNC)
                reward = super().get_reward()
                return np.array([shaping(i, attr.potential, r, attr.done, discounts[i]) if i in discounts else r for i, r in enumerate(reward)])

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            encoding = self.describe()['blob']
            discount = glom.glom(kwargs['config'], 'rl.discount')
            ignore = set(glom.glom(kwargs['config'], 'mdp.pbrs', default=[]))
            setattr(self, PATH_FUNC, {i: glom.glom(kwargs['config'], f"rl.discount_{name}", default=discount) for i, name in enumerate(encoding['reward']) if name not in ignore})
    return MDP
