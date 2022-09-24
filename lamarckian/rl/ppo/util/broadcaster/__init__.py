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

import time
import contextlib
import logging

import numpy as np
import glom
import humanfriendly

import lamarckian


class Fake(object):
    def close(self):
        pass

    def __call__(self, *args, **kwargs):
        pass


class Sync(object):
    def __init__(self, actors, get, **kwargs):
        rpc = getattr(lamarckian.util.rpc.all, glom.glom(kwargs['config'], 'rl.ppo.rpc_all', default=lamarckian.util.rpc.All.__name__))
        self.rpc = rpc(
            actors,
            **{
                'profile': True,
                **kwargs,
                __name__: type(self).__name__,
            }
        )
        self.get = get
        self.fp16 = glom.glom(kwargs['config'], 'rl.ppo.fp16', default=False)
        self.profile = {}

    def close(self):
        return self.rpc.close()

    def __call__(self, *args, **kwargs):
        blob, iteration = self.get()
        self.rpc('set_blob_', [layer.astype(np.float16) for layer in blob] if self.fp16 else blob, iteration)
        self.profile.update(self.rpc.profile)


class Async(lamarckian.rl.remote.Broadcaster):
    def __init__(self, actors, get, *args, **kwargs):
        super().__init__(actors, *args, **{
            'rpc': getattr(lamarckian.util.rpc.all, glom.glom(kwargs['config'], 'rl.ppo.rpc_all', default=lamarckian.util.rpc.All.__name__)),
            'profile': True,
            **kwargs,
            __name__: type(self).__name__,
        })
        self.get = get
        self.fp16 = glom.glom(kwargs['config'], 'rl.ppo.fp16', default=False)
        self.timeout = humanfriendly.parse_timespan(str(glom.glom(self.kwargs['config'], 'rl.ppo.timeout', default=np.iinfo(int).max)))
        self.time = np.finfo(np.float).max
        self.profile = {}

    def __call__(self, *args, **kwargs):
        super().__call__(*args, **kwargs)
        warning = True
        while time.time() - self.time > self.timeout:
            if warning:
                logging.warning('broadcast timeout')
                warning = False
            time.sleep(1)
            super().__call__(*args, **kwargs)

    def broadcast(self, rpc):
        self.time = time.time()
        with contextlib.closing(lamarckian.util.duration.Measure()) as duration:
            blob, iteration = self.get()
        self.profile['blob'] = duration()
        rpc('set_blob_', [layer.astype(np.float16) for layer in blob] if self.fp16 else blob, iteration)
        self.time = np.finfo(np.float).max
        self.profile.update(rpc.profile)


class Hybrid(lamarckian.util.rpc.all.Hybrid):
    def __init__(self, actors, model, get_iteration, *args, **kwargs):
        self.model = model
        self.get_iteration = get_iteration
        super().__init__(actors, *args, **kwargs, **{__name__: type(self).__name__})

    def bind(self):
        import pylamarckian
        RPC = getattr(pylamarckian.rl.ppo.broadcaster, type(self).__name__)
        self.rpc = RPC(
            len(self.branches), self.shards,
            [p.data for p in self.model.parameters()],
            glom.glom(self.kwargs['config'], 'rpc.serializer'),
            glom.glom(self.kwargs['config'], 'rl.ppo.fp16', default=False),
            glom.glom(self.kwargs['config'], 'rpc.sleep', default=0),  # zmq slow joiner
            humanfriendly.parse_timespan(str(glom.glom(self.kwargs['config'], 'rl.ppo.timeout', default=np.iinfo(int).max))),
        )
        return self.rpc.ports

    def wait(self):
        return self.rpc.wait()

    def close(self):
        if hasattr(self, 'rpc'):
            while not self.rpc.stop():
                time.sleep(0.1)
            self.rpc.join()

    def __call__(self):
        return self.rpc(self.get_iteration())

    @property
    def profile(self):
        return self.rpc.profile

    def wrap_optimizer(self, optimizer):
        optimizer.step = self.wrap_step(optimizer.step)

    def wrap_step(self, func):
        def _(*args, **kwargs):
            _ = self.rpc.protect()
            return func(*args, **kwargs)
        return _


class Tree(lamarckian.util.rpc.all.Tree):
    def __init__(self, actors, model, get_iteration, *args, **kwargs):
        self.model = model
        self.get_iteration = get_iteration
        super().__init__(actors, *args, **kwargs, **{__name__: type(self).__name__})

    def bind(self):
        import pylamarckian
        RPC = getattr(pylamarckian.rl.ppo.broadcaster, type(self).__name__)
        self.rpc = RPC(
            len(self.branches),
            [p.data for p in self.model.parameters()],
            glom.glom(self.kwargs['config'], 'rpc.serializer'),
        )
        return self.rpc.ports

    def wait(self):
        return self.rpc.wait()

    def close(self):
        if hasattr(self, 'rpc'):
            return self.rpc.close()

    def __call__(self):
        return self.rpc(self.get_iteration())

    def wrap_optimizer(self, optimizer):
        optimizer.step = self.wrap_step(optimizer.step)

    def wrap_step(self, func):
        def _(*args, **kwargs):
            _ = self.rpc.protect()
            return func(*args, **kwargs)
        return _
