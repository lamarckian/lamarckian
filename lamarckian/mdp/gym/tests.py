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
import contextlib
import itertools
import hashlib
import pickle
import random

import numpy as np
import torch

import lamarckian
import lamarckian.mdp.gym as module

ROOT = os.path.dirname(__file__)


class Agent(object):
    def __init__(self, outputs, action=0):
        self.outputs = outputs
        self.action = action

    def __call__(self, state):
        try:
            return dict(action=torch.LongTensor([self.action]))
        finally:
            self.action = (self.action + 1) % self.outputs


def test_random_process():
    config = lamarckian.util.config.read(ROOT + '.yml')
    sample = 2
    with contextlib.closing(module.MDP(config=config)) as mdp:
        for me in range(len(mdp)):
            encoding = mdp.describe()
            encoding_blob, = (encoding[key] for key in ('blob',))
            assert 'models' in encoding_blob
            mdp.seed(100)
            rs = mdp.get_random_state()
            discrete = encoding_blob['models'][me]['kwargs']['outputs']['discrete']
            digests = (hashlib.md5(pickle.dumps(random.getstate())).hexdigest(), hashlib.md5(pickle.dumps(np.random.get_state())).hexdigest(), hashlib.md5(torch.random.get_rng_state().numpy().tostring()).hexdigest())
            _, results = lamarckian.mdp.evaluate(mdp, me, Agent(discrete), itertools.repeat({}, sample))
            result = mdp.reduce(results)
            _, _results = lamarckian.mdp.evaluate(mdp, me, Agent(discrete), itertools.repeat({}, sample))
            _result = mdp.reduce(_results)
            assert result['fitness'] == _result['fitness'], (result, _result)
            assert hashlib.md5(pickle.dumps(rs)).hexdigest() == hashlib.md5(pickle.dumps(mdp.get_random_state())).hexdigest()
            assert (hashlib.md5(pickle.dumps(random.getstate())).hexdigest(), hashlib.md5(pickle.dumps(np.random.get_state())).hexdigest(), hashlib.md5(torch.random.get_rng_state().numpy().tostring()).hexdigest()) == digests
