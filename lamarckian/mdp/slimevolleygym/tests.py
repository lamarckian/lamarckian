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
import contextlib
import itertools

import torch
import glom

import lamarckian

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


def test():
    config = lamarckian.util.config.read(os.path.dirname(lamarckian.__file__) + '.yml')
    config = lamarckian.util.config.read(ROOT + '.yml', config)
    sample = 2
    MDP = lamarckian.evaluator.parse(*glom.glom(config, 'mdp.create'))
    with contextlib.closing(MDP(config=config)) as mdp:
        for me in range(len(mdp)):
            encoding = mdp.describe()
            encoding_blob, = (encoding[key] for key in ('blob',))
            assert 'models' in encoding_blob
            mdp.seed(100)
            discrete = encoding_blob['models'][me]['kwargs']['outputs']['discrete']
            _, results = lamarckian.mdp.evaluate(mdp, me, Agent(discrete), itertools.repeat({}, sample))
