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

import inspect
import types
import contextlib
import threading
import functools
import traceback
import ctypes
import asyncio

import zmq
try:
    import nnpy
    import pynng
except ImportError:
    pass
import ray.services

from .. import util
from . import map_methods, record
from ..all.wrap import tree as all


def any(actor):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    class Connection(object):
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            if kwargs['backend'] == 'nnpy':
                self.socket = socket = types.SimpleNamespace(args=nnpy.Socket(nnpy.AF_SP, nnpy.PULL), result=nnpy.Socket(nnpy.AF_SP, nnpy.PUSH))
            elif kwargs['backend'] == 'pynng':
                self.socket = socket = types.SimpleNamespace(args=pynng.Pull0(), result=pynng.Push0())
                for s in socket.__dict__.values():
                    s.connect = s.dial
            else:
                self.context = context = zmq.Context()
                self.socket = socket = types.SimpleNamespace(args=context.socket(zmq.PULL), result=context.socket(zmq.PUSH))
            socket.args.connect(kwargs['args'])
            socket.result.connect(kwargs['result'])

        def close(self):
            for socket in self.socket.__dict__.values():
                socket.close()
            if hasattr(self, 'context'):
                self.context.term()

    def run(self, connection):
        asyncio.set_event_loop(asyncio.new_event_loop())
        attr = getattr(self, PATH_FUNC)
        with contextlib.closing(connection):
            while True:
                data = connection.socket.args.recv()
                if data:
                    try:
                        name, args, kwargs = attr.serializer.deserialize(data)
                        if name is None:
                            result, = args
                        else:
                            func = getattr(self, name)
                            result = func(*args, **kwargs)
                        connection.socket.result.send(attr.serializer.serialize(result))
                    except Exception as e:
                        traceback.print_exc()
                        connection.socket.result.send(attr.serializer.serialize(e))
                else:
                    connection.socket.result.send(b'')
                    break

    class Actor(actor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            setattr(self, PATH_FUNC, types.SimpleNamespace(ray=kwargs.get('ray', ray), host=ray.services.get_node_ip_address(), threads=[]))

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
                attr.serializer = util.Serializer(kwargs['serializer'])
                connection = Connection(**kwargs)
                thread = threading.Thread(target=functools.partial(run, self, connection))
                thread.start()
                attr.threads.append(thread)
            else:
                return attr.host
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
            setattr(self, PATH_FUNC, types.SimpleNamespace(ray=kwargs.get('ray', ray), host=ray.services.get_node_ip_address(), connections={}))

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
                attr.serializer = util.Serializer(kwargs['serializer'])
                attr.connections[addr] = connection = types.SimpleNamespace()
                if kwargs['backend'] == 'nnpy':
                    connection.socket = socket = nnpy.Socket(nnpy.AF_SP, nnpy.REQ)
                elif kwargs['backend'] == 'pynng':
                    connection.socket = socket = pynng.Req0()
                    socket.connect = socket.dial
                else:
                    connection.context = context = zmq.Context()
                    connection.socket = socket = context.socket(zmq.REQ)
                socket.connect(addr)
    return Actor
