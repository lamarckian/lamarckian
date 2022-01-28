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

import numpy as np
import glom
import ray

import lamarckian


class Actor(object):
    def __init__(self, index):
        self.index = index

    def echo(self, x=None):
        return x

    def wrong(self):
        raise RuntimeError()


def _test(rpc):
    repr(rpc)
    assert rpc('echo') is None
    result = rpc.fetch_any('echo', np.zeros(1))
    assert isinstance(result, np.ndarray), result
    results = rpc.fetch_all('echo', np.zeros(1))
    for result in results:
        assert isinstance(result, np.ndarray), result
    # wrong
    try:
        rpc('wrong')
        assert False
    except RuntimeError:
        pass
    try:
        rpc.fetch_any('wrong')
        assert False
    except RuntimeError:
        pass
    try:
        rpc.fetch_all('wrong')
        assert False
    except RuntimeError:
        pass


def test_flat():
    from . import Flat as RPC
    from .wrap import flat as wrap
    config = lamarckian.util.config.read(os.path.dirname(lamarckian.__file__) + '.yml')
    ray.init(local_mode=True)
    actor = ray.remote(wrap(Actor))
    parallel = int(ray.cluster_resources()['CPU'])
    actors = [actor.remote(i) for i in range(parallel)]
    with contextlib.closing(RPC(actors, config=config)) as rpc:
        _test(rpc)


def test_tree():
    from . import Tree as RPC
    from .wrap import tree as wrap
    config = lamarckian.util.config.read(os.path.dirname(lamarckian.__file__) + '.yml')
    ray.init(local_mode=True)
    actor = ray.remote(wrap(Actor))
    parallel = int(ray.cluster_resources()['CPU'])
    actors = [actor.remote(i) for i in range(parallel)]
    for key, value in [
        ('rpc.all.local', False),
        ('rpc.all.local', True),
    ]:
        glom.assign(config, key, value, missing=dict)
        with contextlib.closing(RPC(actors, config=config)) as rpc:
            _test(rpc)
