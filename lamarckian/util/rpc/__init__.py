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

import types
import threading
import collections.abc

import zmq
import glom
import filelock
import port_for
import ray

import lamarckian
from . import util, wrap, all
from .all import Tree as All


class Any(object):
    def __init__(self, actors, **kwargs):
        assert actors
        self.actors = actors
        self.kwargs = kwargs
        self.ray = kwargs.get('ray', ray)
        host = self.ray.util.get_node_ip_address()
        with filelock.FileLock(util.LOCK_PORT):
            self.context = zmq.Context()
            self.socket = types.SimpleNamespace(param=self.context.socket(zmq.PUSH), result=self.context.socket(zmq.PULL))
            param_port = port_for.select_random(glom.glom(kwargs['config'], 'rpc.ports', default=None))
            self.socket.param.bind(f"tcp://*:{param_port}")
            result_port = port_for.select_random(glom.glom(kwargs['config'], 'rpc.ports', default=None))
            self.socket.result.bind(f"tcp://*:{result_port}")
        serializer = glom.glom(kwargs['config'], 'rpc.serializer')
        self.serializer = lamarckian.util.serialize.Serializer(kwargs.get('serializer', serializer))
        self.ray.get([actor.setup_rpc_any.remote(param=f"tcp://{host}:{param_port}", result=f"tcp://{host}:{result_port}", **kwargs) for actor in actors])
        for _ in self.actors:
            self.socket.result.recv()
        self.n = 0
        self.lock = threading.Lock()

    def close(self):
        for _ in range(len(self.actors)):
            self.socket.param.send(b'')
        # for _ in self.actors:
        #     self.socket.result.recv()
        for socket in self.socket.__dict__.values():
            socket.close()
        if hasattr(self, 'context'):
            self.context.term()

    def send(self, _, *args, **kwargs):
        return self.socket.param.send(self.serializer.serialize((_, args, kwargs)))

    def receive(self):
        data = self.socket.result.recv()
        result = self.serializer.deserialize(data)
        if isinstance(result, Exception):
            raise result
        return result

    def __call__(self, _, *args, **kwargs):
        data = self.serializer.serialize((_, args, kwargs))
        with self.lock:
            self.socket.param.send(data)
            return self.receive()

    def map(self, tasks):
        assert isinstance(tasks, collections.abc.Iterator)
        with self.lock:
            running = [self.socket.param.send(self.serializer.serialize(task)) for _, task in zip(self.actors, tasks)]
            for task in tasks:
                yield self.receive()
                self.socket.param.send(self.serializer.serialize(task))
            for _ in running:
                yield self.receive()


class Gather(object):
    def __init__(self, rpc_all, **kwargs):
        self.rpc_all = rpc_all
        self.kwargs = kwargs
        self.ray = kwargs.get('ray', ray)
        host = self.ray.util.get_node_ip_address()
        with filelock.FileLock(util.LOCK_PORT):
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REP)
            port = port_for.select_random(glom.glom(kwargs['config'], 'rpc.ports', default=None))
            self.socket.bind(f"tcp://*:{port}")
        self.addr = f"tcp://{host}:{port}"
        serializer = glom.glom(kwargs['config'], 'rpc.serializer')
        self.serializer = lamarckian.util.serialize.Serializer(kwargs.get('serializer', serializer))
        rpc_all('setup_rpc_gather', self.addr, serializer=serializer)

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
                try:
                    self.socket.recv(zmq.NOBLOCK)
                    self.socket.send(b'')
                except zmq.Again:
                    pass
            thread.join()
        return types.SimpleNamespace(close=close)

    def __call__(self):
        data = self.socket.recv()
        self.socket.send(b'')
        result = self.serializer.deserialize(data)
        if isinstance(result, Exception):
            raise result
        else:
            return result
