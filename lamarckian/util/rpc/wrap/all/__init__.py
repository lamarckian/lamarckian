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
import random
import binascii
import struct
import itertools
import contextlib
import datetime
import platform
import logging
import time
from timeit import default_timer as timer

import numpy as np
import zmq
import glom
import filelock
import port_for
import setproctitle
import humanfriendly
import ray

import lamarckian
from lamarckian.util.rpc import util


class Connection(object):
    def __init__(self, threads=1, **kwargs):
        self.kwargs = kwargs
        self.sockets = []
        shm = kwargs.get('shm', 0)
        if shm > 0:
            from multiprocessing import shared_memory
            self.shm = shared_memory.SharedMemory(name=kwargs['shm_name'], create=kwargs.get('shm_create', False), size=shm)
        self.context = zmq.Context(max(threads, 1))

    def connect_upstream(self, param, result):
        self.upstream = socket = types.SimpleNamespace(param=self.context.socket(zmq.SUB), result=self.context.socket(zmq.PUSH))
        socket.param.setsockopt(zmq.SUBSCRIBE, b'')
        socket.param.connect(param)
        socket.result.connect(result)
        self.sockets += list(socket.__dict__.values())
        return param, result

    def bind_downstream(self, param, result, ports=None):
        self.downstream = socket = types.SimpleNamespace(param=self.context.socket(zmq.PUB), result=self.context.socket(zmq.PULL))
        with filelock.FileLock(util.LOCK_PORT):
            if param is None:
                param = f"tcp://*:{port_for.select_random(ports)}"
            socket.param.bind(param)
            if result is None:
                result = f"tcp://*:{port_for.select_random(ports)}"
            socket.result.bind(result)
        self.sockets += list(socket.__dict__.values())
        return param, result

    def close(self):
        for socket in self.sockets:
            socket.close()
        self.context.term()
        if hasattr(self, 'shm'):
            self.shm.close()
            if self.kwargs.get('shm_create', False):
                self.shm.unlink()

    def cache(self, data):
        if hasattr(self, 'shm'):
            size = len(data)
            self.shm.buf[:size] = data
            return size.to_bytes(util.SIZE_BYTES, 'little')
        else:
            return data

    def parse(self, data):
        if hasattr(self, 'shm'):
            size = int.from_bytes(data, 'little')
            data = self.shm.buf[:size]
        return data


def fetch1(results):
    errors = np.array([result[-1] for result in results], np.bool)
    if errors.any():
        indexes, = np.nonzero(errors)
        return results[indexes[0]]
    else:
        return results[0]


def run_router(init, param, result, nodes, index, extra=8 + 1, **kwargs):
    setproctitle.setproctitle(f"rpc_all.router{index}{kwargs.get('suffix', '')}")
    host = ray.util.get_node_ip_address()
    serializer = glom.glom(kwargs['config'], 'rpc.serializer')
    serializer = lamarckian.util.serialize.Serializer(serializer)
    with contextlib.closing(Connection(len(nodes), **kwargs)) as connection:
        connection.connect_upstream(param, result)
        param, result = connection.bind_downstream(None, None, glom.glom(kwargs['config'], 'rpc.ports', default=None))
        init.put((param.replace('*', host), result.replace('*', host)))
        upstream, downstream = connection.upstream, connection.downstream
        time.sleep(glom.glom(kwargs['config'], 'rpc.sleep', default=0))  # zmq slow joiner
        for _ in range(len(nodes)):
            downstream.result.recv()
        upstream.result.send(b'')
        while True:
            data = upstream.param.recv()
            start = timer()
            downstream.param.send(data)
            if data:
                datum = [downstream.result.recv() for _ in range(len(nodes))]
                datum = [data for data in datum if data]
                fetch = util.Fetch(data[-1])
                if fetch == util.Fetch.all:
                    results = list(itertools.chain(*[serializer.deserialize(data) for data in datum]))
                    upstream.result.send(serializer.serialize(results))
                else:
                    if datum:
                        data = fetch1(datum)
                        upstream.result.send(data[:-extra] + struct.pack('<d', timer() - start) + data[-extra:])
                    else:
                        upstream.result.send(b'')
            else:
                for _ in range(len(nodes)):
                    downstream.result.recv()
                upstream.result.send(b'')
                break


def run_leaf(init, param, result, nodes, index, extra=1, **kwargs):
    setproctitle.setproctitle(f"rpc_all.leaf{index}{kwargs.get('suffix', '')}")
    host = ray.util.get_node_ip_address()
    serializer = glom.glom(kwargs['config'], 'rpc.serializer')
    serializer = lamarckian.util.serialize.Serializer(serializer)
    shm_name = binascii.b2a_hex(os.urandom(util.ID_LENGTH)).decode()
    with contextlib.closing(Connection(len(nodes), shm_create=True, shm_name=shm_name, **kwargs)) as connection:
        connection.connect_upstream(param, result)
        param, result = connection.bind_downstream(None, None, glom.glom(kwargs['config'], 'rpc.ports', default=None))
        init.put((param.replace('*', host), result.replace('*', host), shm_name))
        upstream, downstream = connection.upstream, connection.downstream
        time.sleep(glom.glom(kwargs['config'], 'rpc.sleep', default=0))  # zmq slow joiner
        for _ in range(len(nodes)):
            downstream.result.recv()
        upstream.result.send(b'')
        while True:
            data = upstream.param.recv()
            start = timer()
            downstream.param.send(data)
            if data:
                datum = [downstream.result.recv() for _ in range(len(nodes))]
                datum = [data for data in datum if data]
                fetch = util.Fetch(data[-1])
                if fetch == util.Fetch.all:
                    results = [serializer.deserialize(data) for data in datum]
                    upstream.result.send(serializer.serialize(results))
                else:
                    if datum:
                        data = fetch1(datum)
                        upstream.result.send(data[:-extra] + struct.pack('<d', timer() - start) + data[-extra:])
                    else:
                        upstream.result.send(b'')
            else:
                for _ in range(len(nodes)):
                    downstream.result.recv()
                upstream.result.send(b'')
                break
