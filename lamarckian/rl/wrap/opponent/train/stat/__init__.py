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
import collections
import types
import operator
import logging

import numpy as np
import glom


class Win(object):
    def __init__(self, sample):
        assert sample > 0, sample
        self.history = collections.deque(maxlen=sample)

    def add(self, win):
        self.history.append(win)

    def __float__(self):
        if self.history:
            return np.mean(self.history)
        else:
            return 0.0

    def __len__(self):
        return len(self.history)

    def __repr__(self):
        return f'{float(self)}={sum(self.history)}/{len(self)}|{self.history.maxlen}'


def win(rl):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    def drop_loser(opponents, capacity):
        if len(opponents) >= capacity:
            indexes, _ = zip(*sorted([(i, float(opponent[NAME_FUNC])) for i, opponent in enumerate(opponents.values())], key=operator.itemgetter(1))[capacity - 1:])
            digests = list(opponents.keys())
            for i in indexes:
                digest = digests[i]
                del opponents[digest]

    def __setstate__(self, wins):
        opponents = self._get_opponents_train().values()
        assert len(wins) == len(opponents), (len(wins), len(opponents))
        for win, opponent in zip(wins, opponents):
            if isinstance(win, Win):
                opponent[NAME_FUNC] = win
            elif isinstance(win, collections.deque):
                opponent[NAME_FUNC] = Win(win.maxlen)
                opponent[NAME_FUNC].history = win

    class RL(rl):
        def __init__(self, state={}, *args, **kwargs):
            super().__init__(state, *args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            setattr(self, PATH_FUNC, types.SimpleNamespace(
                capacity=glom.glom(kwargs['config'], 'rl.opponent.train.pool.capacity', default=np.iinfo(np.int).max),
                sample=glom.glom(self.kwargs['config'], f'rl.opponent.train.stat.{NAME_FUNC}.sample'),
                drop_loser=glom.glom(self.kwargs['config'], f'rl.opponent.train.stat.{NAME_FUNC}.drop_loser', default=False),
                debug=glom.glom(kwargs['config'], f'rl.opponent.train.stat.{NAME_FUNC}.debug', default=False),
            ))
            if PATH_FUNC in state:
                __setstate__(self, state[PATH_FUNC])

        def set_opponents_train(self, *args, **kwargs):
            try:
                return super().set_opponents_train(*args, **kwargs)
            finally:
                attr = getattr(self, PATH_FUNC)
                for opponent in self._get_opponents_train().values():
                    assert NAME_FUNC not in opponent
                    opponent[NAME_FUNC] = Win(attr.sample)

        def append_opponent_train(self, *args, **kwargs):
            attr = getattr(self, PATH_FUNC)
            if attr.drop_loser:
                drop_loser(self._get_opponents_train(), attr.capacity)
            try:
                return super().append_opponent_train(*args, **kwargs)
            finally:
                opponent = next(reversed(self._get_opponents_train().values()))
                assert NAME_FUNC not in opponent
                opponent[NAME_FUNC] = Win(attr.sample)

        def __call__(self, *args, **kwargs):
            attr = getattr(self, PATH_FUNC)
            outcome = super().__call__(*args, **kwargs)
            opponents = self._get_opponents_train()
            for result in outcome['results']:
                digest = result['digest_opponent_train']
                if digest in opponents:
                    opponent = opponents[digest]
                    opponent[NAME_FUNC].add(result[NAME_FUNC])
                elif attr.debug:
                    logging.warning(f'opponent {digest} not in opponents_train')
            return outcome

        def __getstate__(self):
            state = super().__getstate__()
            assert PATH_FUNC not in state
            state[PATH_FUNC] = [opponent[NAME_FUNC].history for opponent in self._get_opponents_train().values()]
            return state

        def __setstate__(self, state):
            state = super().__setstate__(state)
            __setstate__(self, state[PATH_FUNC])
            return state
    return RL
