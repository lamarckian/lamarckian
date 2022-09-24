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
import types
import itertools
import collections
import copy
import random
import inspect
import logging

import numpy as np
import glom

import lamarckian

NAME = os.path.basename(os.path.dirname(__file__))


def choose(rl):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    class RL(rl):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            setattr(self, PATH_FUNC, types.SimpleNamespace(
                prob=glom.glom(kwargs['config'], f'rl.opponent.train.{NAME}.prob'),
                random=random.Random(glom.glom(kwargs['config'], 'seed', default=None)),
            ))

        def _choose_opponent_train(self):
            attr = getattr(self, PATH_FUNC)
            styles = getattr(self, f'{__name__}.{NAME}').styles
            style = attr.random.choice(list(styles.values()))
            if attr.random.random() < attr.prob:
                return style.pool[-1]
            else:
                items_former = list(style.pool)[:-1]
                if items_former:
                    return attr.random.choice(items_former)
                else:
                    return style.pool[-1]
    return RL


def sync(rl):
    PATH_FUNC = f'{__name__}.{NAME}'

    def switch(self):
        attr = getattr(self, PATH_FUNC)
        blob = copy.deepcopy(self.get_blob())
        attr.style.pool.append(dict(blobs={key: blob for key in attr.style.pool[-1]['blobs']}))
        style, attr.style = next(attr.iter)
        self.set_hparam(dict(value=attr.style.hparam))
        blob = next(iter(attr.style.pool[-1]['blobs'].values()))
        self.set_blob(blob)
        self.set_opponent_train(self._choose_opponent_train()['blobs'])
        logging.info(f'switch to model {style}')

    class RL(rl):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            blob = copy.deepcopy(super().get_blob())
            maxlen = glom.glom(kwargs['config'], 'rl.opponent.train.pool.capacity', default=np.iinfo(np.int).max)
            weight = glom.glom(kwargs['config'], f'rl.opponent.train.{NAME}.weight')
            keys = set(itertools.chain(*map(list, weight.values())))
            styles = {key: types.SimpleNamespace(
                hparam={key: hparam.get(key, 0) for key in keys},
                pool=collections.deque([dict(
                    blobs={enemy: blob for enemy in self.enemies},
                )], maxlen=maxlen),
            ) for key, hparam in weight.items()}
            assert len(styles) > 1, weight
            Stopper = lamarckian.evaluator.parse(*glom.glom(kwargs['config'], f"rl.opponent.train.{NAME}.stopper"), **kwargs)
            make_stopper = lambda: Stopper(self, **kwargs)
            attr = types.SimpleNamespace(
                make_stopper=make_stopper,
                stopper=make_stopper(),
                styles=styles,
                iter=itertools.cycle(styles.items()),
            )
            setattr(self, PATH_FUNC, attr)
            style, attr.style = next(attr.iter)

        def training(self):
            switch(self)
            return super().training()

        def __call__(self, *args, **kwargs):
            outcome = super().__call__(*args, **kwargs)
            attr = getattr(self, PATH_FUNC)
            if attr.stopper(outcome):
                attr.stopper = attr.make_stopper()
                switch(self)
            return outcome

        def __setstate__(self, state):
            attr = getattr(self, PATH_FUNC)
            maxlen, = {style.pool.maxlen for style in attr.styles}
            attr.styles = state['styles']
            for style in attr.styles.values():
                style.pool = collections.deque(style.pool, maxlen=maxlen)
            attr.iter = itertools.cycle(attr.styles.items())
            return super().__setstate__(state)

        def __getstate__(self):
            state = super().__getstate__()
            assert not hasattr(state, 'styles')
            attr = getattr(self, PATH_FUNC)
            state['styles'] = attr.styles
            return state
    return RL
