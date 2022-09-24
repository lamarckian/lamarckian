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
import tempfile

import torch.utils.tensorboard

import lamarckian
from . import Rollout

PREFIX = os.path.splitext(__file__)[0]


def test_rollout():
    config = lamarckian.util.config.read(PREFIX + '.yml')
    kwargs = dict(config=config)
    Evaluator = lamarckian.evaluator.parse(*config['evaluator']['create'], config=config)
    evaluator = Evaluator(config=config)
    with tempfile.TemporaryDirectory() as root, torch.utils.tensorboard.SummaryWriter(os.path.join(root, 'log')) as writer:
        recorder = types.SimpleNamespace(evaluator=evaluator, writer=writer, put=lambda record: record(recorder))
        individual = dict(decision=evaluator.get())
        recorder.put(Rollout('rollout/result', 0, [individual]))
