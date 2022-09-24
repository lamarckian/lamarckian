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
import contextlib
import functools

import tqdm
import ray

import lamarckian

PREFIX = os.path.splitext(__file__)[0]


def test_global_best():
    config = lamarckian.util.config.read(os.path.join(PREFIX, inspect.getframeinfo(inspect.currentframe()).function[len('test_'):] + '.yml'))
    ray.init(local_mode=True)
    EC = lamarckian.evaluator.parse(*config['ec']['create'], config=config)
    with contextlib.closing(EC(config=config)) as ea:
        for _ in tqdm.trange(100):
            ea()
    ray.shutdown()
    assert ea.elite['result']['fitness'] > 186
