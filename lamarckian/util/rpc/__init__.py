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

import types
import threading
import collections.abc

import zmq
try:
    import nnpy
    import pynng
except ImportError:
    pass
import glom
import filelock
import port_for
import ray.services
import ray.exceptions

from . import util, wrap, all
from .all import Tree as All


class Any(object):
    def __init__(self, actors, **kwargs):
        assert actors
        self.actors = actors
        self.kwargs = kwargs
        self.ray = kwargs.get('ray', ray)
        host = self.ray.services.get_node_ip_address()
        backend = glom.glom(kwargs['config'], 'rpc.backend')
        with filelock.FileLock(util.LOCK_PORT):
            if backend == 'nnpy':
                self.socket = types.SimpleNamespace(args=nnpy.Socket(nnpy.AF_SP, nnpy.PUSH), result=nnpy.Socket(nnpy.AF_SP, nnpy.PULL))
            elif backend == 'pynng':
                self.socket = types.SimpleNamespace(args=pynng.Push0(), result=pynng.Pull0())
                for socket in self.socket.__dict__.values():
                    socket.bind = socket.listen
            else:
                self.context = zmq.Context()
                self.socket = types.SimpleNamespace(args=self.context.socket(zmq.PUSH), result=self.context.socket(zmq.PULL))
            self.port_args = port_for.select_random(glom.glom(kwargs['config'], 'rpc.ports', default=None))
            self.socket.args.bind(f"tcp://*:{self.port_args}")
            self.port_result = port_for.select_random(glom.glom(kwargs['config'], 'rpc.ports', default=None))
            self.socket.result.bind(f"tcp://*:{self.port_result}")
        serializer = glom.glom(kwargs['config'], 'rpc.serializer')
        self.serializer = util.Serializer(kwargs.get('serializer', serializer))
        self.ray.get([actor.setup_rpc_any.remote(args=f"tcp://{host}:{self.port_args}", result=f"tcp://{host}:{self.port_result}", backend=backend, serializer=serializer) for actor in actors])
        self.n = 0
        self.lock = threading.Lock()

    def close(self):
        for _ in range(len(self.actors)):
            self.socket.args.send(b'')
        # for _ in range(len(self.actors)):
        #     self.socket.result.recv()
        for socket in self.socket.__dict__.values():
            socket.close()
        if hasattr(self, 'context'):
            self.context.term()

    def send(self, _, *args, **kwargs):
        return self.socket.args.send(self.serializer.serialize((_, args, kwargs)))

    def receive(self):
        data = self.socket.result.recv()
        result = self.serializer.deserialize(data)
        if isinstance(result, Exception):
            raise result
        return result

    def __call__(self, _, *args, **kwargs):
        data = self.serializer.serialize((_, args, kwargs))
        with self.lock:
            self.socket.args.send(data)
            return self.receive()

    def map(self, tasks):
        assert isinstance(tasks, collections.abc.Iterator)
        with self.lock:
            running = [self.socket.args.send(self.serializer.serialize(task)) for _, task in zip(self.actors, tasks)]
            for task in tasks:
                yield self.receive()
                self.socket.args.send(self.serializer.serialize(task))
            for _ in running:
                yield self.receive()


class Gather(object):
    def __init__(self, rpc_all, **kwargs):
        self.rpc_all = rpc_all
        self.kwargs = kwargs
        self.ray = kwargs.get('ray', ray)
        host = self.ray.services.get_node_ip_address()
        self.backend = glom.glom(kwargs['config'], 'rpc.backend')
        with filelock.FileLock(util.LOCK_PORT):
            if self.backend == 'nnpy':
                self.socket = nnpy.Socket(nnpy.AF_SP, nnpy.REP)
            elif self.backend == 'pynng':
                self.socket = pynng.Rep0()
                self.socket.bind = self.socket.listen
            else:
                self.context = zmq.Context()
                self.socket = self.context.socket(zmq.REP)
            port = port_for.select_random(glom.glom(kwargs['config'], 'rpc.ports', default=None))
            self.socket.bind(f"tcp://*:{port}")
        self.addr = f"tcp://{host}:{port}"
        serializer = glom.glom(kwargs['config'], 'rpc.serializer')
        self.serializer = util.Serializer(kwargs.get('serializer', serializer))
        rpc_all('setup_rpc_gather', self.addr, backend=self.backend, serializer=serializer)

    def close(self):
        self.socket.close()
        if hasattr(self, 'context'):
            self.context.term()

    def __len__(self):
        return len(self.rpc_all)

    def gathering(self, name):
        self.rpc_all('setup_rpc_gather', self.addr, name=name)

        def close():
            thread = threading.Thread(target=lambda: self.rpc_all('setup_rpc_gather', self.addr, name=None))
            thread.start()
            while thread.is_alive():
                if self.backend == 'nnpy':
                    try:
                        self.socket.recv(nnpy.DONTWAIT)
                        self.socket.send(b'')
                    except nnpy.NNError:
                        pass
                elif self.backend == 'pynng':
                    try:
                        self.socket.recv(block=False)
                        self.socket.send(b'0')
                    except pynng.exceptions.TryAgain:
                        pass
                else:
                    try:
                        self.socket.recv(zmq.NOBLOCK)
                        self.socket.send(b'')
                    except zmq.Again:
                        pass
            thread.join()
        return types.SimpleNamespace(close=close)

    def __call__(self):
        data = self.socket.recv()
        self.socket.send(b'0')  # for pynng
        result = self.serializer.deserialize(data)
        if isinstance(result, Exception):
            raise result
        else:
            return result
