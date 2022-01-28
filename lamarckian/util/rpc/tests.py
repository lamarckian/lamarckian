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

import tqdm
import ray

import lamarckian


class Actor(object):
    def __init__(self, index):
        self.index = index

    def echo(self, x=None):
        return x

    def wrong(self):
        raise RuntimeError()


def test_any():
    config = lamarckian.util.config.read(os.path.dirname(lamarckian.__file__) + '.yml')
    ray.init(local_mode=True)
    actor = ray.remote(lamarckian.util.rpc.wrap.any(Actor))
    parallel = int(ray.cluster_resources()['CPU'])
    actors = [actor.remote(i) for i in range(parallel)]
    for _ in tqdm.trange(3):
        with contextlib.closing(lamarckian.util.rpc.Any(actors, config=config)) as rpc:
            rpc('echo')
            try:
                rpc('wrong')
                assert False
            except RuntimeError:
                pass


def test_gather():
    config = lamarckian.util.config.read(os.path.dirname(lamarckian.__file__) + '.yml')
    ray.init(local_mode=True)
    actor = ray.remote(lamarckian.util.rpc.wrap.gather(lamarckian.util.rpc.wrap.all(Actor)))
    parallel = int(ray.cluster_resources()['CPU'])
    actors = [actor.remote(i) for i in range(parallel)]
    for _ in tqdm.trange(3):
        with contextlib.closing(lamarckian.util.rpc.All(actors, config=config)) as rpc_all, contextlib.closing(lamarckian.util.rpc.Gather(rpc_all, config=config)) as rpc:
            for _ in tqdm.trange(3):
                with contextlib.closing(rpc.gathering('echo')):
                    for _ in range(3):
                        rpc()
            for _ in tqdm.trange(3):
                try:
                    with contextlib.closing(rpc.gathering('wrong')):
                        rpc()
                    assert False
                except RuntimeError:
                    pass
