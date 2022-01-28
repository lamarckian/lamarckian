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
import copy
import hashlib
import pickle
import random

import numpy as np
import torch

import lamarckian
from lamarckian.rl.pg import Evaluator
from .. import CODING
from . import Split as Crossover


def test_same():
    config = lamarckian.util.config.read(os.path.dirname(lamarckian.__file__) + '.yml')
    evaluator = Evaluator(config=config)
    crossover = Crossover(evaluator.describe()[CODING], config=config, **{**config['crossover'][CODING], **dict(prob=1)})
    parent = evaluator.initialize()[CODING]
    digests = (hashlib.md5(pickle.dumps(random.getstate())).hexdigest(), hashlib.md5(pickle.dumps(np.random.get_state())).hexdigest(), hashlib.md5(torch.random.get_rng_state().numpy().tostring()).hexdigest())
    child1, child2 = crossover(copy.deepcopy(parent), copy.deepcopy(parent))
    for child in (child1, child2):
        for p, c in zip(parent, child):
            np.testing.assert_array_almost_equal(p, c)
    assert (hashlib.md5(pickle.dumps(random.getstate())).hexdigest(), hashlib.md5(pickle.dumps(np.random.get_state())).hexdigest(), hashlib.md5(torch.random.get_rng_state().numpy().tostring()).hexdigest()) == digests


def test_clone():
    config = lamarckian.util.config.read(os.path.dirname(lamarckian.__file__) + '.yml')
    evaluator = Evaluator(config=config)
    crossover = Crossover(evaluator.describe()[CODING], config=config, **{**config['crossover'][CODING], **dict(prob=1)})
    ancestor = evaluator.initialize()[CODING], evaluator.initialize()[CODING]
    digest = hashlib.md5(pickle.dumps(ancestor)).hexdigest()
    digests = (hashlib.md5(pickle.dumps(random.getstate())).hexdigest(), hashlib.md5(pickle.dumps(np.random.get_state())).hexdigest(), hashlib.md5(torch.random.get_rng_state().numpy().tostring()).hexdigest())
    offspring = crossover(*ancestor)
    for child in offspring:
        for p1, p2, c in zip(*ancestor, child):
            assert p1.shape == c.shape
            assert p2.shape == c.shape
    assert hashlib.md5(pickle.dumps(ancestor)).hexdigest() == digest
    assert (hashlib.md5(pickle.dumps(random.getstate())).hexdigest(), hashlib.md5(pickle.dumps(np.random.get_state())).hexdigest(), hashlib.md5(torch.random.get_rng_state().numpy().tostring()).hexdigest()) == digests
