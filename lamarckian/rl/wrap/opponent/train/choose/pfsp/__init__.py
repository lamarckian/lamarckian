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

import lamarckian


def draw(rl):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    class RL(rl):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            setattr(self, PATH_FUNC, types.SimpleNamespace(
                scale=eval('lambda x: ' + glom.glom(kwargs['config'], 'rl.opponent.train.pfsp.scale')),
                sample=glom.glom(kwargs['config'], 'rl.opponent.train.pfsp.sample', default=9),
            ))

        def get_pfsp_probs(self, opponents):
            attr = getattr(self, PATH_FUNC)
            win = np.array([float(opponent['win']) for opponent in opponents])
            probs = np.abs(win - 0.5)
            probs = probs.max() - probs
            probs = attr.scale(probs)
            mask = np.array([len(opponent['win']) for opponent in opponents]) < attr.sample
            if mask.any():
                probs[mask] = probs.max()
            return lamarckian.util.to_probs(probs)
    return RL
