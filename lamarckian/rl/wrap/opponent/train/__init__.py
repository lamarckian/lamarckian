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

import glom

import lamarckian
from . import choose, pool, debug


def ensure(rl):
    class RL(rl):
        def get_opponent_train(self):
            opponent = super().get_opponent_train()
            assert opponent
            return opponent
    return RL


def trigger(rl):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    class Broadcaster(lamarckian.rl.remote.Broadcaster):
        def __init__(self, rl, *args, **kwargs):
            super().__init__(rl.actors, *args, **rl.kwargs, **kwargs)
            self.rl = rl

        def broadcast(self, rpc):
            blobs = self.rl._choose_opponent_train()['blobs']
            assert blobs, blobs
            rpc('set_opponent_train', blobs)

    class RL(rl):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            setattr(self, PATH_FUNC, types.SimpleNamespace(
                broadcaster=Broadcaster(self),
                stopper=eval('lambda outcome: ' + str(glom.glom(kwargs['config'], f"rl.opponent.train.{NAME_FUNC}", default=True))),
            ))

        def __call__(self, *args, **kwargs):
            attr = getattr(self, PATH_FUNC)
            outcome = super().__call__(*args, **kwargs)
            if attr.stopper(outcome):
                attr.broadcaster()
            return outcome
    return RL


def file(section='rl.opponent.train.file'):
    from .. import file
    return file(section, 'opponents_train')
