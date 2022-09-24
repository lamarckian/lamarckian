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
import time

import torch
import torch.nn as nn
import glom
import ray

import lamarckian


class Actor(object):
    def __init__(self, index):
        self.index = index
        self.model = nn.Linear(3, 2)
        self.iteration = 0

    def set_blob(self, blob, iteration):
        self.iteration = iteration


def test_hybrid():
    from lamarckian.rl.ppo.util.broadcaster import Hybrid as RPC
    from lamarckian.rl.ppo.wrap.rpc import all as wrap
    config = lamarckian.util.config.read(os.path.dirname(lamarckian.__file__) + '.yml')
    glom.assign(config, 'ray.local_mode', True, missing=dict)
    ray.init(local_mode=True)
    actor = ray.remote(wrap(Actor))
    parallel = 1  # int(ray.cluster_resources()['CPU'])
    actors = [actor.remote(i) for i in range(parallel)]
    model = nn.Linear(3, 2)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    for modifies in [
        [('rpc.all.fake', 3)],
        [('rpc.all.fake', 0)],
        [('rpc.shm', '1m')],
    ]:
        for key, value in modifies:
            glom.assign(config, key, value, missing=dict)
        with contextlib.closing(RPC(actors, model, lambda: 0, config=config)) as rpc:
            model.load_state_dict({key: torch.rand(*value.shape, dtype=value.dtype, device=value.device) for key, value in model.state_dict().items()})
            rpc()
            time.sleep(1)
