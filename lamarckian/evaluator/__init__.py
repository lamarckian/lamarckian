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
import types
import random
import logging

import numpy as np
import torch
import glom
import ray

import lamarckian
from . import wrap, record
from typing import Dict

class Evaluator(object):
    """
    Base class for evolutionary/RL algorithms.

    :param state: The run-time state to resume from.
    """
    def __init__(self, state={}, **kwargs):
        self.kwargs = kwargs
        self.cost = state.get('cost', 0)
        self.ray = kwargs.get('ray', ray)

    def close(self):
        pass

    def __len__(self):
        return 1

    def seed(self, seed) -> None:
        pass

    def describe(self) -> Dict:
        return {}

    def initialize(self) -> Dict:
        """Initialize the algorithm, e.g., setup hyperparameters and generate initial population."""
        return {}

    def set(self, decision: Dict) -> None:
        """
        :param decision: The decision (solution) that uniquely specifies an algorithm instance.
        """
        pass

    def get(self) -> Dict:
        return {}

    def training(self):
        """
        Setup a context that will be used by contextlib.closing() to maanage the resources.
        """
        return types.SimpleNamespace(close=lambda: None)

    def __call__(self):
        """
        Main execution process of the algorithm.
        """
        raise NotImplementedError()

    def evaluate(self):
        """
        Evaluate an individual policy/the current population.
        """
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

    def __call__(self):
        if hasattr(self, 'root'):
            os.makedirs(self.root, exist_ok=True)
            path = os.path.join(self.root, f'{self.evaluator.cost}.pth')
            logging.info(f"save {path}")
            torch.save(self.evaluator.__getstate__(), path)
            if self.keep > 0:
                lamarckian.util.file.tidy(self.root, self.keep)
            else:
                logging.warning(f'keep all models in {self.root}')

    @staticmethod
    def counter(**kwargs):
        return lamarckian.util.counter.Time(**glom.glom(kwargs['config'], 'record.save'))
