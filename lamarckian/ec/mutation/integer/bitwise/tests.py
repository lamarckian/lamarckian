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
import random

import numpy as np
import torch

import lamarckian
from .. import CODING
from . import Mutation


def test_clone():
    config = lamarckian.util.config.read(os.path.dirname(lamarckian.__file__) + '.yml')
    encoding = np.array([(-3000, 5000), (-1000, 4000)], np.int)
    mutation = Mutation(encoding, config=config, **{**config['mutation'][CODING], **dict(prob=1)})
    parent = np.array([np.random.randint(lower, upper) for lower, upper in zip(*encoding.T)])
    digest = hashlib.md5(pickle.dumps(parent)).hexdigest()
    digests = (hashlib.md5(pickle.dumps(random.getstate())).hexdigest(), hashlib.md5(pickle.dumps(np.random.get_state())).hexdigest(), hashlib.md5(torch.random.get_rng_state().numpy().tostring()).hexdigest())
    mutation(parent)
    assert hashlib.md5(pickle.dumps(parent)).hexdigest() == digest
    assert (hashlib.md5(pickle.dumps(random.getstate())).hexdigest(), hashlib.md5(pickle.dumps(np.random.get_state())).hexdigest(), hashlib.md5(torch.random.get_rng_state().numpy().tostring()).hexdigest()) == digests
