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

import os
import inspect
import logging
import types
import collections.abc
import functools
import pickle
import copy
import time
import hashlib
import traceback

import numpy as np
import torch
import glom

import lamarckian


NAME = os.path.basename(os.path.dirname(__file__))


def flat(rl):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    def load(self, path):
        attr = getattr(self, PATH_FUNC)
        if path.endswith('.pth'):
            state = torch.load(os.path.expandvars(os.path.expanduser(path)))
            return glom.glom(state, attr.spec)
        else:
            return path

    class RL(rl):
        def __init__(self, state={}, *args, **kwargs):
            super().__init__(state, *args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            spec = glom.glom(kwargs['config'], f'rl.opponent.train.spec', default='decision.blob')
            spec = [int(key) if key.lstrip('-+').isdigit() else key for key in spec.split('.')]
            spec = functools.reduce(lambda T, key: T[key], [glom.T] + spec)
            capacity = glom.glom(kwargs['config'], 'rl.opponent.train.pool.capacity', default=np.iinfo(np.int).max)
            if capacity <= 0:
                capacity = np.iinfo(np.int).max
            setattr(self, PATH_FUNC, types.SimpleNamespace(
                capacity=capacity,
                opponents=collections.OrderedDict([]),
                spec=spec,
            ))
            if 'opponents_train' in state:
                RL.set_opponents_train(self, state['opponents_train'])

        def training(self):
            attr = getattr(self, PATH_FUNC)
            size = glom.glom(self.kwargs['config'], 'rl.opponent.train.trim', default=np.iinfo(np.int).max)
            if size < len(attr.opponents):
                attr.opponents = collections.OrderedDict(list(reversed(list(reversed(attr.opponents.items()))[:size])))
            for path in glom.glom(self.kwargs['config'], 'rl.opponent.train.transfer', default=[]):
                blob = load(self, path)
                self.append_opponent_train({enemy: blob for enemy in self.enemies})
            attr.opponents = collections.OrderedDict(list(reversed(list(reversed(attr.opponents.items()))[:attr.capacity])))
            if not attr.opponents:
                blob = copy.deepcopy(self.get_blob())
                self.set_opponents_train([{enemy: blob for enemy in self.enemies}])
            self.set_opponent_train(self._choose_opponent_train()['blobs'])
            return super().training()

        def _get_opponents_train(self):
            return getattr(self, PATH_FUNC).opponents

        def get_opponents_train(self):
            return [item['blobs'] for key, item in getattr(self, PATH_FUNC).opponents.items()]

        def set_opponents_train(self, opponents, **kwargs):
            assert isinstance(opponents, collections.abc.Iterable), type(opponents)
            attr = getattr(self, PATH_FUNC)
            attr.opponents = collections.OrderedDict([(hashlib.md5(pickle.dumps(list(blobs.values()))).hexdigest(), {**kwargs, **dict(blobs=blobs)}) for blobs in opponents])

        def append_opponent_train(self, blobs, **kwargs):
            assert blobs and all(isinstance(index, int) for index in blobs)
            attr = getattr(self, PATH_FUNC)
            digest = hashlib.md5(pickle.dumps(list(blobs.values()))).hexdigest()
            attr.opponents[digest] = {**kwargs, **dict(blobs=blobs)}
            if len(attr.opponents) > attr.capacity:
                attr.opponents = collections.OrderedDict(list(attr.opponents.items())[-attr.capacity:])

        def append_opponents_train(self, opponents, **kwargs):
            for opponent in opponents:
                self.append_opponent_train(opponent, **kwargs)

        def __getstate__(self):
            state = super().__getstate__()
            if glom.glom(self.kwargs['config'], 'rl.opponent.train.save', default=True):
                state['opponents_train'] = self.get_opponents_train()
            return state

        def __setstate__(self, state):
            if 'opponents_train' in state:
                self.set_opponents_train(state['opponents_train'])
            return super().__setstate__(state)

        def _choose_opponent_train(self):
            raise NotImplementedError()
    return RL


def append(rl):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    class Stat(object):
        def __init__(self, **kwargs):
            self.cost = 0
            self.iteration = 0
            self.episode = 0
            self.results = collections.deque(maxlen=glom.glom(kwargs['config'], 'sample.train'))
            self.time = time.time()

        @property
        def result(self):
            return lamarckian.util.reduce(self.results)

    @flat
    class RL(rl):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            setattr(self, PATH_FUNC, types.SimpleNamespace(
                stat=Stat(**kwargs),
                stopper=eval('lambda self, outcome, stat: ' + str(glom.glom(kwargs['config'], f"rl.opponent.train.pool.{NAME_FUNC}"))),
            ))
            if hasattr(self, 'recorder'):
                self.recorder.register(
                    lamarckian.util.counter.Time(**glom.glom(kwargs['config'], 'record.scalar')),
                    lambda *args, **kwargs: lamarckian.util.record.Scalar(self.cost, **{'self_play/opponents': len(self._get_opponents_train())}),
                )

        def __call__(self, *args, **kwargs):
            cost = self.cost
            outcome = super().__call__(*args, **kwargs)
            attr = getattr(self, PATH_FUNC)
            stat = attr.stat
            stat.cost += self.cost - cost
            stat.iteration += 1
            stat.episode += len(outcome['results'])
            stat.results += outcome['results']
            if attr.stopper(self, outcome, stat):
                attr.stat = Stat(**self.kwargs)
                blob = copy.deepcopy(super().get_blob())
                opponent = {enemy: blob for enemy in self.enemies}
                self.append_opponent_train(opponent)
            return outcome
    return RL


def watch_dir(rl):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    class RL(rl):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            Stopper = lamarckian.evaluator.parse(*glom.glom(kwargs['config'], f"rl.opponent.train.pool.{NAME_FUNC}.stopper"), **kwargs)
            make_stopper = lambda: Stopper(self, **kwargs)
            spec = glom.glom(kwargs['config'], f'rl.opponent.train.pool.{NAME_FUNC}.spec', default='decision.blob')
            spec = [int(key) if key.lstrip('-+').isdigit() else key for key in spec.split('.')]
            spec = functools.reduce(lambda T, key: T[key], [glom.T] + spec)
            setattr(self, PATH_FUNC, types.SimpleNamespace(
                make_stopper=make_stopper,
                stopper=make_stopper(),
                spec=spec,
                root=os.path.expandvars(os.path.expanduser(glom.glom(kwargs['config'], f'rl.opponent.train.pool.{NAME_FUNC}.root'))),
            ))

        def __call__(self, *args, **kwargs):
            attr = getattr(self, PATH_FUNC)
            outcome = super().__call__(*args, **kwargs)
            if attr.stopper(outcome):
                attr.stopper = attr.make_stopper()
                try:
                    for filename in os.listdir(attr.root):
                        if filename.endswith('.pth'):
                            path = os.path.join(attr.root, filename)
                            state = torch.load(path)
                            blob = glom.glom(state, attr.spec)
                            opponent = {enemy: blob for enemy in self.enemies}
                            self.append_opponent_train(opponent)
                            os.remove(path)
                            logging.info(f'append opponent in {path}')
                except:
                    traceback.print_exc()
            return outcome
    return RL
