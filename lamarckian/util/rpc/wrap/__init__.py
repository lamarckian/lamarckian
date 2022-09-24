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
import types
import contextlib
import multiprocessing
import threading
import queue
import functools
import traceback
import ctypes
import signal
import asyncio

import zmq
import glom
import ray

import lamarckian
from . import map_methods, record
from .. import util


def all(actor):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    def execute(self, serializer, name, args, kwargs, fetch=False):
        try:
            func = getattr(self, name)
            result = func(*args, **kwargs)
            if fetch:
                return serializer.serialize(result) + bytes([0])
            else:
                return bytes([0])
        except Exception as e:
            traceback.print_exc()
            return serializer.serialize(e) + bytes([1])

    def run(self, param, result, **kwargs):
        from .all import Connection
        asyncio.set_event_loop(asyncio.new_event_loop())
        serializer = glom.glom(kwargs['config'], 'rpc.serializer')
        serializer = lamarckian.util.serialize.Serializer(serializer)
        with contextlib.closing(Connection(**kwargs)) as connection:
            connection.connect_upstream(param, result)
            upstream = connection.upstream
            upstream.result.send(b'')
            while True:
                data = connection.parse(upstream.param.recv())
                if data:
                    (name, args, kwargs), fetch = serializer.deserialize(data[:-1]), util.Fetch(data[-1])
                    upstream.result.send(execute(self, serializer, name, args, kwargs, fetch != util.Fetch.call))
                else:
                    del data
                    upstream.result.send(b'')
                    break

    # setup

    def setup_execute(*args, **kwargs):
        threading.Thread(target=run, args=args, kwargs=kwargs).start()

    def setup_router(self, *args, **kwargs):
        from .all import run_router as run
        if glom.glom(kwargs['config'], 'ray.local_mode', default=False):
            init = queue.Queue()
            threading.Thread(target=run, args=(init,) + args, kwargs=kwargs).start()
            return init.get()
        else:
            init = multiprocessing.Queue()
            pid = os.fork()
            if pid:
                return init.get()
            else:
                try:
                    import prctl
                    prctl.set_pdeathsig(signal.SIGKILL)
                except ImportError:
                    traceback.print_exc()
                run(init, *args, **kwargs)

    def setup_assembler(self, *args, **kwargs):
        from .all import run_assembler as run
        if glom.glom(kwargs['config'], 'ray.local_mode', default=False):
            init = queue.Queue()
            threading.Thread(target=run, args=(init,) + args, kwargs=kwargs).start()
            return init.get()
        else:
            init = multiprocessing.Queue()
            pid = os.fork()
            if pid:
                return init.get()
            else:
                try:
                    import prctl
                    prctl.set_pdeathsig(signal.SIGKILL)
                except ImportError:
                    traceback.print_exc()
                run(init, *args, **kwargs)

    def setup_leaf(self, *args, **kwargs):
        from .all import run_leaf as run
        if glom.glom(kwargs['config'], 'ray.local_mode', default=False):
            init = queue.Queue()
            threading.Thread(target=run, args=(init,) + args, kwargs=kwargs).start()
            return init.get()
        else:
            init = multiprocessing.Queue()
            pid = os.fork()
            if pid:
                return init.get()
            else:
                try:
                    import prctl
                    prctl.set_pdeathsig(signal.SIGKILL)
                except ImportError:
                    traceback.print_exc()
                run(init, *args, **kwargs)

    setup = dict(
        execute=setup_execute,
        router=setup_router,
        assembler=setup_assembler,
        leaf=setup_leaf,
    )

    class Actor(actor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)

        def setup_rpc_all(self, mode=None, *args, **kwargs):
            if mode is None:
                return ray.util.get_node_ip_address()
            return setup[mode](self, *args, **kwargs)
    return Actor


def any(actor):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    class Connection(object):
        def __init__(self, **kwargs):
            self.context = context = zmq.Context()
            self.socket = socket = types.SimpleNamespace(param=context.socket(zmq.PULL), result=context.socket(zmq.PUSH))
            socket.param.connect(kwargs['param'])
            socket.result.connect(kwargs['result'])

        def close(self):
            for socket in self.socket.__dict__.values():
                socket.close()
            if hasattr(self, 'context'):
                self.context.term()

    def run(self, **kwargs):
        asyncio.set_event_loop(asyncio.new_event_loop())
        serializer = glom.glom(kwargs['config'], 'rpc.serializer')
        serializer = lamarckian.util.serialize.Serializer(serializer)
        with contextlib.closing(Connection(**kwargs)) as connection:
            socket = connection.socket
            socket.result.send(b'')
            while True:
                data = socket.param.recv()
                if data:
                    try:
                        name, args, kwargs = serializer.deserialize(data)
                        if name is None:
                            result, = args
                        else:
                            func = getattr(self, name)
                            result = func(*args, **kwargs)
                        socket.result.send(serializer.serialize(result))
                    except Exception as e:
                        traceback.print_exc()
                        socket.result.send(serializer.serialize(e))
                else:
                    socket.result.send(b'')
                    break

    class Actor(actor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            setattr(self, PATH_FUNC, types.SimpleNamespace(ray=kwargs.get('ray', ray), threads=[]))

        def close(self):
            attr = getattr(self, PATH_FUNC)
            for thread in attr.threads:
                try:
                    id = next((id for id, t in threading._active.items() if t is thread))
                    ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(id), ctypes.py_object(SystemExit))
                    thread.join()
                except StopIteration:
                    pass
            return super().close()

        def setup_rpc_any(self, **kwargs):
            attr = getattr(self, PATH_FUNC)
            if kwargs:
                thread = threading.Thread(target=run, args=(self,), kwargs=kwargs)
                thread.start()
                attr.threads.append(thread)
            else:
                return ray.util.get_node_ip_address()
    return Actor


def any_count(rpc):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    class RPC(rpc):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            setattr(self, PATH_FUNC, 0)

        def send(self, *args, **kwargs):
            setattr(self, PATH_FUNC, getattr(self, PATH_FUNC) + 1)
            return super().send(*args, **kwargs)

        def receive(self):
            setattr(self, PATH_FUNC, getattr(self, PATH_FUNC) - 1)
            return super().receive()

        def __len__(self):
            return getattr(self, PATH_FUNC)

        def __call__(self, *args, **kwargs):
            assert not getattr(self, PATH_FUNC)
            return super().__call__(*args, **kwargs)

        def map(self, *args, **kwargs):
            assert not getattr(self, PATH_FUNC)
            return super().map(*args, **kwargs)
    return RPC


def gather(actor):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    def run(self, connection, name):
        asyncio.set_event_loop(asyncio.new_event_loop())
        attr = getattr(self, PATH_FUNC)
        while connection.running:
            try:
                func = getattr(self, name)
                result = func()
                connection.socket.send(attr.serializer.serialize(result))
            except Exception as e:
                traceback.print_exc()
                connection.socket.send(attr.serializer.serialize(e))
            connection.socket.recv()

    class Actor(actor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            setattr(self, PATH_FUNC, types.SimpleNamespace(ray=kwargs.get('ray', ray), host=ray.util.get_node_ip_address(), connections={}))

        def close(self):
            attr = getattr(self, PATH_FUNC)
            for connection in attr.connections.values():
                connection.socket.close()
                if hasattr(connection, 'context'):
                    connection.context.term()
            return super().close()

        def setup_rpc_gather(self, addr, **kwargs):
            attr = getattr(self, PATH_FUNC)
            if 'name' in kwargs:
                if kwargs['name'] is None:
                    connection = attr.connections[addr]
                    connection.running = False
                    connection.thread.join()
                else:
                    connection = attr.connections[addr]
                    connection.running = True
                    connection.thread = threading.Thread(target=functools.partial(run, self, connection, kwargs['name']))
                    connection.thread.start()
            else:
                assert addr not in attr.connections, addr
                attr.serializer = lamarckian.util.serialize.Serializer(kwargs['serializer'])
                attr.connections[addr] = connection = types.SimpleNamespace()
                connection.context = context = zmq.Context()
                connection.socket = socket = context.socket(zmq.REQ)
                socket.connect(addr)
    return Actor
