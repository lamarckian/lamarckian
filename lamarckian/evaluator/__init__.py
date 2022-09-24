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
import contextlib
import logging

import torch
import glom
import ray

import lamarckian
from . import wrap, record


class Evaluator(object):
    def __init__(self, state={}, **kwargs):
        self.kwargs = kwargs
        self.cost = state.get('cost', 0)
        self.ray = kwargs.get('ray', ray)

    def close(self):
        pass

    def __len__(self):
        return 1

    def seed(self, seed):
        pass

    def describe(self):
        return {}

    def initialize(self):
        return {}

    def set(self, decision):
        pass

    def get(self):
        return {}

    def training(self):
        return types.SimpleNamespace(close=lambda: None)

    def __call__(self):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()

    def reduce(self, *args, **kwargs):
        return lamarckian.util.reduce(*args, **kwargs)

    def __setstate__(self, state):
        self.cost = state['cost']
        self.set(state['decision'])
        return state

    def __getstate__(self):
        return dict(cost=self.cost, decision=self.get(), encoding=self.describe())


def parse(cls, *args, **kwargs):
    if isinstance(cls, str):
        cls = lamarckian.util.parse.instance(cls)
    for wrap in args:
        if isinstance(wrap, str):
            wrap = lamarckian.util.parse.instance(wrap)
        if inspect.getfullargspec(wrap).varkw:
            cls = wrap(cls, **kwargs)
        else:
            cls = wrap(cls)
    return cls


class Saver(object):
    def __init__(self, evaluator):
        self.evaluator = evaluator
        kwargs = evaluator.kwargs
        root = kwargs.get('root', None)
        if root is not None and 'index' not in kwargs:
            self.root = os.path.expanduser(os.path.expandvars(root))
            self.keep = glom.glom(kwargs['config'], 'model.keep', default=0)

    def close(self):
        self()

    def __call__(self):
        if hasattr(self, 'root'):
            os.makedirs(self.root, exist_ok=True)
            path = os.path.join(self.root, f'{self.evaluator.cost}.pth')
            if not os.path.exists(path):
                with contextlib.closing(lamarckian.util.DelayNewline()) as logger:
                    logging.info(f"save {path}")
                    state = self.evaluator.__getstate__()
                    logger(' ... ')
                    torch.save(state, path)
                    logger('done')
                if self.keep > 0:
                    lamarckian.util.file.tidy(self.root, self.keep)
                else:
                    logging.warning(f'keep all models in {self.root}')

    @staticmethod
    def counter(**kwargs):
        return lamarckian.util.counter.Time(**glom.glom(kwargs['config'], 'record.save'))
