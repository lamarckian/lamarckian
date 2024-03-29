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
import hashlib
import pickle
import copy
import random

import numpy as np
import torch

import lamarckian
from .. import CODING
from . import Crossover


def test_same():
    config = lamarckian.util.config.read(os.path.dirname(lamarckian.__file__) + '.yml')
    encoding = np.array([(-3000, 5000), (-1000, 4000)], np.int)
    crossover = Crossover(encoding, config=config, **{**config['crossover'][CODING], **dict(prob=1)})
    parent = np.array([np.random.randint(lower, upper) for lower, upper in zip(*encoding.T)])
    digests = (hashlib.md5(pickle.dumps(random.getstate())).hexdigest(), hashlib.md5(pickle.dumps(np.random.get_state())).hexdigest(), hashlib.md5(torch.random.get_rng_state().numpy().tostring()).hexdigest())
    offspring = crossover(copy.deepcopy(parent), copy.deepcopy(parent))
    for child in offspring:
        np.testing.assert_array_almost_equal(parent, child)
    assert (hashlib.md5(pickle.dumps(random.getstate())).hexdigest(), hashlib.md5(pickle.dumps(np.random.get_state())).hexdigest(), hashlib.md5(torch.random.get_rng_state().numpy().tostring()).hexdigest()) == digests


def test_clone():
    config = lamarckian.util.config.read(os.path.dirname(lamarckian.__file__) + '.yml')
    encoding = np.array([(-3000, 5000), (-1000, 4000)], np.int)
    crossover = Crossover(encoding, config=config, **{**config['crossover'][CODING], **dict(prob=1)})
    ancestor = np.array([np.random.randint(lower, upper) for lower, upper in zip(*encoding.T)]), np.array([np.random.randint(lower, upper) for lower, upper in zip(*encoding.T)])
    digest = hashlib.md5(pickle.dumps(ancestor)).hexdigest()
    digests = (hashlib.md5(pickle.dumps(random.getstate())).hexdigest(), hashlib.md5(pickle.dumps(np.random.get_state())).hexdigest(), hashlib.md5(torch.random.get_rng_state().numpy().tostring()).hexdigest())
    crossover(*ancestor)
    assert hashlib.md5(pickle.dumps(ancestor)).hexdigest() == digest
    assert (hashlib.md5(pickle.dumps(random.getstate())).hexdigest(), hashlib.md5(pickle.dumps(np.random.get_state())).hexdigest(), hashlib.md5(torch.random.get_rng_state().numpy().tostring()).hexdigest()) == digests
