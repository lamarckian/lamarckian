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
import inspect
import types
import enum
import contextlib
import threading
import functools
import itertools
import binascii
import traceback
import asyncio

import numpy as np
import zmq
try:
    import nnpy
    import pynng
except ImportError:
    pass
import filelock
import port_for
import ray.services

from ... import util

Fetch = enum.Enum('Fetch', zip('call any all'.split(), itertools.count()))


def flat(actor):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    class Connection(object):
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.context = context = zmq.Context()
            self.socket = socket = types.SimpleNamespace(args=context.socket(zmq.SUB), result=context.socket(zmq.PUSH))
            socket.args.setsockopt(zmq.SUBSCRIBE, b'')
            socket.args.connect(kwargs['args'])
            socket.result.connect(kwargs['result'])

        def close(self):
            for socket in self.socket.__dict__.values():
                socket.close()
            if hasattr(self, 'context'):
                self.context.term()

    def execute(self, name, args, kwargs, fetch=False):
        attr = getattr(self, PATH_FUNC)
        try:
            func = getattr(self, name)
            result = func(*args, **kwargs)
            if fetch:
                return attr.serializer.serialize(result) + bytes([0])
            else:
                return bytes([0])
        except Exception as e:
            traceback.print_exc()
            return attr.serializer.serialize(e) + bytes([1])

    def run(self, **kwargs):
        attr = getattr(self, PATH_FUNC)
        asyncio.set_event_loop(asyncio.new_event_loop())
        with contextlib.closing(Connection(**kwargs)) as connection:
            socket = connection.socket
            while True:
                data = socket.args.recv()
                if data:
                    (name, args, kwargs), fetch = attr.serializer.deserialize(data[:-1]), Fetch(data[-1])
                    socket.result.send(execute(self, name, args, kwargs, fetch != Fetch.call))
                else:
                    socket.result.send(b'')
                    break

    class Actor(actor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            setattr(self, PATH_FUNC, types.SimpleNamespace(ray=kwargs.get('ray', ray), host=ray.services.get_node_ip_address()))

        def setup_rpc_all(self, **kwargs):
            attr = getattr(self, PATH_FUNC)
            if kwargs:
                attr.serializer = util.Serializer(kwargs['serializer'])
                attr.thread = threading.Thread(target=functools.partial(run, self, **kwargs))
                attr.thread.start()
            else:
                return attr.host
    return Actor


def tree(actor):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    class Connection(object):
        def __init__(self, create, **kwargs):
            self.create = create
            self.kwargs = kwargs
            if kwargs['shm'] > 0:
                from multiprocessing import shared_memory
                self.shm = shared_memory.SharedMemory(name=kwargs['shm_name'], create=create, size=kwargs['shm'])
            if kwargs['backend'] == 'nnpy':
                self.socket = socket = types.SimpleNamespace(args=nnpy.Socket(nnpy.AF_SP, nnpy.SUB), result=nnpy.Socket(nnpy.AF_SP, nnpy.PUSH))
                socket.args.setsockopt(nnpy.SUB, nnpy.SUB_SUBSCRIBE, '')
            elif kwargs['backend'] == 'pynng':
                self.socket = socket = types.SimpleNamespace(args=pynng.Sub0(), result=pynng.Push0())
                for s in socket.__dict__.values():
                    s.connect = s.dial
                socket.args.subscribe(b'')
            else:
                self.context = context = zmq.Context()
                self.socket = socket = types.SimpleNamespace(args=context.socket(zmq.SUB), result=context.socket(zmq.PUSH))
                socket.args.setsockopt(zmq.SUBSCRIBE, b'')
            socket.args.connect(kwargs['args'])
            socket.result.connect(kwargs['result'])

        def close(self):
            for socket in self.socket.__dict__.values():
                socket.close()
            if hasattr(self, 'context'):
                self.context.term()
            if hasattr(self, 'shm'):
                self.shm.close()
                if self.create:
                    self.shm.unlink()

        def cache(self, data):
            if hasattr(self, 'shm'):
                size = len(data)
                self.shm.buf[:size] = data
                return size.to_bytes(4, 'big')
            else:
                return data

        def parse(self, data):
            if hasattr(self, 'shm'):
                size = int.from_bytes(data, 'big')
                data = self.shm.buf[:size]
            return data

    def execute(self, name, args, kwargs, fetch=False):
        attr = getattr(self, PATH_FUNC)
        try:
            func = getattr(self, name)
            result = func(*args, **kwargs)
            if fetch:
                return attr.serializer.serialize(result) + bytes([0])
            else:
                return bytes([0])
        except Exception as e:
            traceback.print_exc()
            return attr.serializer.serialize(e) + bytes([1])

    def fetch1(results):
        errors = np.array([result[-1] for result in results], np.bool)
        if errors.any():
            indexes, = np.nonzero(errors)
            return results[indexes[0]]
        else:
            return results[0]

    def run_execute(self, connection):
        asyncio.set_event_loop(asyncio.new_event_loop())
        attr = getattr(self, PATH_FUNC)
        socket = connection.socket
        while True:
            data = connection.parse(socket.args.recv())
            if data:
                (name, args, kwargs), fetch = attr.serializer.deserialize(data[:-1]), Fetch(data[-1])
                socket.result.send(execute(self, name, args, kwargs, fetch != Fetch.call))
            else:
                socket.result.send(b'')
                connection.close()
                break

    def run_node(self, connection, actors):
        asyncio.set_event_loop(asyncio.new_event_loop())
        attr = getattr(self, PATH_FUNC)
        socket = connection.socket
        while True:
            data = socket.args.recv()
            socket.args_execute.send(connection.cache(data))
            if data:
                (name, args, kwargs), fetch = attr.serializer.deserialize(data[:-1]), Fetch(data[-1])
                results = [execute(self, name, args, kwargs, fetch != Fetch.call)] + [socket.result_execute.recv() for _ in range(actors)]
                socket.result.send(attr.serializer.serialize(results))
            else:
                for _ in range(actors):
                    socket.result_execute.recv()
                socket.result.send(b'')
                connection.close()
                break

    def run_relay(self, connection, actors, nodes):
        asyncio.set_event_loop(asyncio.new_event_loop())
        attr = getattr(self, PATH_FUNC)
        socket = connection.socket
        while True:
            data = socket.args.recv()
            socket.args_execute.send(connection.cache(data))
            socket.args_relay.send(data)
            if data:
                (name, args, kwargs), fetch = attr.serializer.deserialize(data[:-1]), Fetch(data[-1])
                results = [execute(self, name, args, kwargs, fetch != Fetch.call)] + [socket.result_execute.recv() for _ in range(actors)] + list(itertools.chain(*[attr.serializer.deserialize(socket.result_relay.recv()) for _ in range(nodes)]))
                if fetch == Fetch.all:
                    socket.result.send(attr.serializer.serialize([attr.serializer.deserialize(result[:-1]) for result in results]))
                else:
                    socket.result.send(fetch1(results))
            else:
                for _ in range(actors):
                    socket.result_execute.recv()
                for _ in range(nodes):
                    socket.result_relay.recv()
                socket.result.send(b'')
                connection.close()
                break

    def bind_execute(connection, **kwargs):
        socket = connection.socket
        if kwargs['actors']:
            addr_args, addr_result = f"ipc://{util.IPC_PREFIX}{binascii.b2a_hex(os.urandom(util.ID_LENGTH)).decode()}", f"ipc://{util.IPC_PREFIX}{binascii.b2a_hex(os.urandom(util.ID_LENGTH)).decode()}"
            if kwargs['backend'] == 'nnpy':
                socket.args_execute, socket.result_execute = nnpy.Socket(nnpy.AF_SP, nnpy.PUB), nnpy.Socket(nnpy.AF_SP, nnpy.PULL)
            elif kwargs['backend'] == 'pynng':
                _ = socket.args_execute, socket.result_execute = pynng.Pub0(), pynng.Pull0()
                for s in _:
                    s.bind = s.listen
            else:
                socket.args_execute, socket.result_execute = connection.context.socket(zmq.PUB), connection.context.socket(zmq.PULL)
            socket.args_execute.bind(addr_args)
            socket.result_execute.bind(addr_result)
            return addr_args, addr_result
        else:
            socket.args_execute = types.SimpleNamespace(send=lambda *args, **kwargs: None, close=lambda: None)
            return None, None

    def bind_relay(connection, **kwargs):
        socket = connection.socket
        if kwargs['nodes']:
            if kwargs['backend'] == 'nnpy':
                socket.args_relay, socket.result_relay = nnpy.Socket(nnpy.AF_SP, nnpy.PUB), nnpy.Socket(nnpy.AF_SP, nnpy.PULL)
            elif kwargs['backend'] == 'pynng':
                _ = socket.args_relay, socket.result_relay = pynng.Pub0(), pynng.Pull0()
                for s in _:
                    s.bind = s.listen
            else:
                socket.args_relay, socket.result_relay = connection.context.socket(zmq.PUB), connection.context.socket(zmq.PULL)
            with filelock.FileLock(util.LOCK_PORT):
                port_args = port_for.select_random(kwargs['ports'])
                socket.args_relay.bind(f"tcp://*:{port_args}")
                port_result = port_for.select_random(kwargs['ports'])
                socket.result_relay.bind(f"tcp://*:{port_result}")
            return port_args, port_result
        else:
            socket.args_relay = types.SimpleNamespace(send=lambda *args, **kwargs: None, close=lambda: None)
            return None, None

    def setup_relay(self, **kwargs):
        attr = getattr(self, PATH_FUNC)
        kwargs['shm_name'] = binascii.b2a_hex(os.urandom(util.ID_LENGTH)).decode()
        connection = Connection(True, **kwargs)
        addr_args, addr_result = bind_execute(connection, **kwargs)
        port_args, port_result = bind_relay(connection, **kwargs)
        # run
        actors, nodes = kwargs['actors'], kwargs['nodes']
        attr.thread = threading.Thread(target=functools.partial(run_relay, self, connection, len(actors), len(nodes)))
        attr.thread.start()
        attr.ray.get([getattr(actor, 'setup_rpc_all').remote(
            'execute',
            args=addr_args, result=addr_result,
            backend=kwargs['backend'], serializer=kwargs['serializer'], shm=kwargs['shm'], shm_name=kwargs['shm_name'],
        ) for actor in actors] + [getattr(node[0], 'setup_rpc_all').remote(
            'node',
            args=f"tcp://{attr.host}:{port_args}", result=f"tcp://{attr.host}:{port_result}",
            actors=node[1:],
            backend=kwargs['backend'], serializer=kwargs['serializer'], shm=kwargs['shm'],
        ) for node in nodes])

    def setup_node(self, **kwargs):
        attr = getattr(self, PATH_FUNC)
        kwargs['shm_name'] = binascii.b2a_hex(os.urandom(util.ID_LENGTH)).decode()
        connection = Connection(True, **kwargs)
        addr_args, addr_result = bind_execute(connection, **kwargs)
        # run
        actors = kwargs['actors']
        attr.thread = threading.Thread(target=functools.partial(run_node, self, connection, len(actors)))
        attr.thread.start()
        attr.ray.get([getattr(actor, 'setup_rpc_all').remote(
            'execute',
            args=addr_args, result=addr_result,
            backend=kwargs['backend'], serializer=kwargs['serializer'], shm=kwargs['shm'], shm_name=kwargs['shm_name'],
        ) for actor in actors])

    def setup_execute(self, **kwargs):
        attr = getattr(self, PATH_FUNC)
        connection = Connection(False, **kwargs)
        attr.thread = threading.Thread(target=functools.partial(run_execute, self, connection))
        attr.thread.start()

    class Actor(actor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            setattr(self, PATH_FUNC, types.SimpleNamespace(ray=kwargs.get('ray', ray), host=ray.services.get_node_ip_address()))

        def setup_rpc_all(self, mode=None, **kwargs):
            attr = getattr(self, PATH_FUNC)
            if mode is None:
                return attr.host
            attr.serializer = util.Serializer(kwargs['serializer'])
            if mode == 'relay':
                setup_relay(self, **kwargs)
            elif mode == 'node':
                setup_node(self, **kwargs)
            elif mode == 'execute':
                setup_execute(self, **kwargs)
            else:
                assert False, mode
    return Actor
