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
import types
import random
import numbers

import numpy as np
import glom
import imagecodecs
import cv2

import lamarckian
from . import reward, fitness, objective, image, stat, opponent, debug


def cast(index, dtype=np.float32):
    def decorate(mdp):
        class MDP(mdp):
            class Controller(mdp.Controller):
                def get_state(self):
                    state = super().get_state()
                    state['inputs'][index] = state['inputs'][index].astype(dtype)
                    return state
        return MDP
    return decorate


def expand_dims(index, *args, **kwargs):
    def decorate(mdp):
        class MDP(mdp):
            class Controller(mdp.Controller):
                def get_state(self):
                    state = super().get_state()
                    state['inputs'][index] = np.expand_dims(state['inputs'][index], *args, **kwargs)
                    return state

            def describe_blob(self):
                encoding = super().describe_blob()
                for model in encoding['models']:
                    model['inputs'][index]['shape'] = np.expand_dims(np.zeros(model['inputs'][index]['shape']), *args, **kwargs).shape
                return encoding
        return MDP
    return decorate


def skip(mdp):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    class MDP(mdp):
        class Controller(mdp.Controller):
            async def __call__(self, *args, **kwargs):
                attr = getattr(self.mdp, PATH_FUNC)
                reward = []
                for _ in range(attr.skip):
                    exp = await super().__call__(*args, **kwargs)
                    reward.append(super().get_reward())
                    if exp['done']:
                        break
                setattr(self, PATH_FUNC, lamarckian.rl.cumulate(reward, attr.discount)[0])
                return exp

            def get_reward(self):
                return getattr(self, PATH_FUNC)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            skip = glom.glom(kwargs['config'], 'mdp.skip')
            if isinstance(skip, numbers.Integral):
                lower = upper = skip
            else:
                lower, upper = skip
            assert lower <= upper, (lower, upper)
            r = random.Random(glom.glom(kwargs['config'], 'seed', default=None))
            setattr(self, PATH_FUNC, types.SimpleNamespace(
                get_skip=lambda: r.randint(lower, upper),
                discount=glom.glom(kwargs['config'], 'rl.discount'),
            ))

        def evaluating(self, *args, **kwargs):
            attr = getattr(self, PATH_FUNC)
            get_skip = attr.get_skip
            attr.get_skip = lambda: 1
            evaluating = super().evaluating(*args, **kwargs)

            def close():
                attr.get_skip = get_skip
                return evaluating.close()
            return types.SimpleNamespace(close=close)

        def reset(self, *args, **kwargs):
            attr = getattr(self, PATH_FUNC)
            attr.skip = attr.get_skip()
            return super().reset(*args, **kwargs)
    return MDP


def _skip(mdp):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    class MDP(mdp):
        class Controller(mdp.Controller):
            async def __call__(self, *args, **kwargs):
                attr = getattr(self.mdp, PATH_FUNC)
                for _ in range(attr.skip):
                    exp = await super().__call__(*args, **kwargs)
                    if exp['done']:
                        break
                return exp

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            skip = glom.glom(kwargs['config'], 'mdp.skip')
            if isinstance(skip, numbers.Integral):
                lower = upper = skip
            else:
                lower, upper = skip
            assert lower <= upper, (lower, upper)
            r = random.Random(glom.glom(kwargs['config'], 'seed', default=None))
            setattr(self, PATH_FUNC, types.SimpleNamespace(
                get_skip=lambda: r.randint(lower, upper),
            ))

        def evaluating(self, *args, **kwargs):
            attr = getattr(self, PATH_FUNC)
            get_skip = attr.get_skip
            attr.get_skip = lambda: 1
            evaluating = super().evaluating(*args, **kwargs)

            def close():
                attr.get_skip = get_skip
                return evaluating.close()
            return types.SimpleNamespace(close=close)

        def reset(self, *args, **kwargs):
            attr = getattr(self, PATH_FUNC)
            attr.skip = attr.get_skip()
            return super().reset(*args, **kwargs)
    return MDP


def render(mdp):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    class MDP(mdp):
        class Controller(mdp.Controller):
            def get_state(self):
                attr = getattr(self.mdp, PATH_FUNC)
                state = super().get_state()
                if self.me == attr.me:
                    image = self.mdp.render()
                    if isinstance(image, np.ndarray):
                        state['image'] = imagecodecs.jpeg_encode(image.copy())
                        if attr.show:
                            try:
                                cv2.imshow('g78', image)
                                cv2.waitKey(1)
                            except:
                                pass
                return state

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            setattr(self, PATH_FUNC, types.SimpleNamespace(
                me=glom.glom(kwargs['config'], 'rl.me', default=0),
                show=glom.glom(kwargs['config'], 'mdp.render.show', default=False),
            ))
    return MDP
