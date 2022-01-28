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
import itertools
import operator
import types
import binascii
import threading

import zmq
try:
    import nnpy
    import pynng
except ImportError:
    pass
import glom
import toolz.itertoolz
import humanfriendly
import filelock
import port_for
import ray.services
import ray.exceptions

import lamarckian
from .. import util
from .wrap import Fetch
from . import wrap, tree

NL = '\n'
NLNL = '\n\n'
TABLE_HEADER = '| node | actors |\n| :-- | :-- |\n'


class Flat(object):
    def __init__(self, actors, **kwargs):
        assert actors
        self.kwargs = kwargs
        self.ray = kwargs.get('ray', ray)
        try:
            tasks = [actor.setup_rpc_all.remote() for actor in actors]
            desc = kwargs.get('progress', None)
            if desc is not None:
                lamarckian.util.remote.progress(tasks, desc, self.ray)
            self.actors = {actor: host for actor, host in zip(actors, self.ray.get(tasks))}
        except ray.exceptions.RayActorError:
            raise lamarckian.util.remote.RayActorError(f"{', '.join(util.detect_died_actors(actors, lambda actor: actor.setup_rpc_all.remote()))} died")
        self.context = zmq.Context()
        self.socket = types.SimpleNamespace(args=self.context.socket(zmq.PUB), result=self.context.socket(zmq.PULL))
        host = ray.services.get_node_ip_address()
        ports = glom.glom(kwargs['config'], 'rpc.ports', default=None)
        with filelock.FileLock('/tmp/lamarckian.port'):
            port_args = port_for.select_random(ports)
            self.socket.args.bind(f"tcp://*:{port_args}")
            port_result = port_for.select_random(ports)
            self.socket.result.bind(f"tcp://*:{port_result}")
        # run
        serializer = glom.glom(kwargs['config'], 'rpc.serializer')
        self.serializer = util.Serializer(kwargs.get('serializer', serializer))
        self.ray.get([actor.setup_rpc_all.remote(
            args=f"tcp://{host}:{port_args}", result=f"tcp://{host}:{port_result}", ports=ports,
            serializer=serializer,
        ) for actor in actors])
        self.lock = threading.Lock()

    def close(self):
        self.socket.args.send(b'')
        for _ in range(len(self.actors)):
            self.socket.result.recv()
        for socket in self.socket.__dict__.values():
            socket.close()
        if hasattr(self, 'context'):
            self.context.term()

    def __len__(self):
        return len(self.actors)

    def __call__(self, _, *args, **kwargs):
        data = self.serializer.serialize((_, args, kwargs)) + bytes([Fetch.call.value])
        with self.lock:
            self.socket.args.send(data)
            results = [self.socket.result.recv() for _ in range(len(self.actors))]
        for data in results:
            if data[-1]:
                raise self.serializer.deserialize(data[:-1])

    def fetch_any(self, _, *args, **kwargs):
        data = self.serializer.serialize((_, args, kwargs)) + bytes([Fetch.any.value])
        with self.lock:
            self.socket.args.send(data)
            results = [self.socket.result.recv() for _ in range(len(self.actors))]
        for data in results:
            if data[-1]:
                raise self.serializer.deserialize(data[:-1])
        return self.serializer.deserialize(data[:-1])

    def fetch_all(self, _, *args, **kwargs):
        data = self.serializer.serialize((_, args, kwargs)) + bytes([Fetch.all.value])
        with self.lock:
            self.socket.args.send(data)
            results = [self.serializer.deserialize(self.socket.result.recv()[:-1]) for _ in range(len(self.actors))]
        assert len(results) == len(self.actors), (len(results), len(self.actors))
        for result in results:
            if isinstance(result, Exception):
                raise result
        return results

    def __repr__(self):
        nodes = {host: list(map(operator.itemgetter(0), grouped)) for host, grouped in toolz.groupby(operator.itemgetter(1), self.actors.items()).items()}
        return TABLE_HEADER + NL.join([f"| {host} | {len(actors)} |" for host, actors in nodes.items()])


class Tree(object):
    def __init__(self, actors, **kwargs):
        assert actors
        self.actors = actors
        self.kwargs = kwargs
        self.ray = kwargs.get('ray', ray)
        try:
            tasks = [actor.setup_rpc_all.remote() for actor in actors]
            desc = kwargs.get('progress', None)
            if desc is not None:
                lamarckian.util.remote.progress(tasks, desc, self.ray)
            self.topology = tree.Topology(
                {actor: host for actor, host in zip(actors, self.ray.get(tasks))},
                degree=glom.glom(kwargs['config'], 'rpc.all.degree', default=None),
                local=glom.glom(kwargs['config'], 'rpc.all.local', default=True),
            )
            # socket
            backend = glom.glom(kwargs['config'], 'rpc.backend')
            if backend == 'nnpy':
                self.local = types.SimpleNamespace(args=nnpy.Socket(nnpy.AF_SP, nnpy.PUB), result=nnpy.Socket(nnpy.AF_SP, nnpy.PULL))
                self.remote = types.SimpleNamespace(args=nnpy.Socket(nnpy.AF_SP, nnpy.PUB), result=nnpy.Socket(nnpy.AF_SP, nnpy.PULL))
            elif backend == 'pynng':
                self.local = types.SimpleNamespace(args=pynng.Pub0(), result=pynng.Pull0())
                self.remote = types.SimpleNamespace(args=pynng.Pub0(), result=pynng.Pull0())
                for socket in itertools.chain(self.local.__dict__.values(), self.remote.__dict__.values()):
                    socket.bind = socket.listen
            else:
                self.context = zmq.Context()
                self.local = types.SimpleNamespace(args=self.context.socket(zmq.PUB), result=self.context.socket(zmq.PULL))
                self.remote = types.SimpleNamespace(args=self.context.socket(zmq.PUB), result=self.context.socket(zmq.PULL))
            # bind
            addr_args, addr_result = f"ipc://{util.IPC_PREFIX}{binascii.b2a_hex(os.urandom(util.ID_LENGTH)).decode()}", f"ipc://{util.IPC_PREFIX}{binascii.b2a_hex(os.urandom(util.ID_LENGTH)).decode()}"
            self.local.args.bind(addr_args)
            self.local.result.bind(addr_result)
            ports = glom.glom(kwargs['config'], 'rpc.ports', default=None)
            with filelock.FileLock('/tmp/lamarckian.port'):
                port_args = port_for.select_random(ports)
                self.remote.args.bind(f"tcp://*:{port_args}")
                port_result = port_for.select_random(ports)
                self.remote.result.bind(f"tcp://*:{port_result}")
            # run
            serializer = glom.glom(kwargs['config'], 'rpc.serializer')
            self.serializer = util.Serializer(kwargs.get('serializer', serializer))
            shm = humanfriendly.parse_size(str(glom.glom(kwargs['config'], 'rpc.shm', default=0)))
            shm_name = binascii.b2a_hex(os.urandom(util.ID_LENGTH)).decode()
            if shm > 0:
                from multiprocessing import shared_memory
                self.shm = shared_memory.SharedMemory(name=shm_name, create=True, size=shm)
            self.ray.get([actor.setup_rpc_all.remote(
                'execute',
                args=addr_args, result=addr_result,
                backend=backend, serializer=serializer, shm=shm, shm_name=shm_name,
            ) for actor in self.topology.local] + [node[0].setup_rpc_all.remote(
                'relay',
                args=f"tcp://{self.topology.host}:{port_args}", result=f"tcp://{self.topology.host}:{port_result}", ports=ports,
                actors=node[1:], nodes=nodes,
                backend=backend, serializer=serializer, shm=shm,
            ) for node, nodes in self.topology.relays.items()])
        except ray.exceptions.RayActorError:
            raise lamarckian.util.remote.RayActorError(f"{', '.join(util.detect_died_actors(actors, lambda actor: actor.setup_rpc_all.remote()))} died")
        self.lock = threading.Lock()

    def close(self):
        if self.topology.local:
            self.local.args.send(b'')
        if self.topology.relays:
            self.remote.args.send(b'')
        for _ in range(len(self.topology.local)):
            self.local.result.recv()
        for _ in range(len(self.topology.relays)):
            self.remote.result.recv()
        for socket in itertools.chain(self.local.__dict__.values(), self.remote.__dict__.values()):
            socket.close()
        if hasattr(self, 'context'):
            self.context.term()
        if hasattr(self, 'shm'):
            self.shm.close()
            self.shm.unlink()

    def __len__(self):
        return len(self.actors)

    def cache(self, data):
        if hasattr(self, 'shm'):
            size = len(data)
            self.shm.buf[:size] = data
            return size.to_bytes(4, 'big')
        else:
            return data

    def __call__(self, _, *args, **kwargs):
        data = self.serializer.serialize((_, args, kwargs)) + bytes([Fetch.call.value])
        with self.lock:
            if self.topology.local:
                self.local.args.send(self.cache(data))
            if self.topology.relays:
                self.remote.args.send(data)
            results = [self.local.result.recv() for _ in range(len(self.topology.local))]
            results += [self.remote.result.recv() for _ in range(len(self.topology.relays))]
        for data in results:
            if data[-1]:
                raise self.serializer.deserialize(data[:-1])

    def fetch_any(self, _, *args, **kwargs):
        data = self.serializer.serialize((_, args, kwargs)) + bytes([Fetch.any.value])
        with self.lock:
            if self.topology.local:
                self.local.args.send(self.cache(data))
            if self.topology.relays:
                self.remote.args.send(data)
            results = [self.local.result.recv() for _ in range(len(self.topology.local))]
            results += [self.remote.result.recv() for _ in range(len(self.topology.relays))]
        for data in results:
            if data[-1]:
                raise self.serializer.deserialize(data[:-1])
        return self.serializer.deserialize(data[:-1])

    def fetch_all(self, _, *args, **kwargs):
        data = self.serializer.serialize((_, args, kwargs)) + bytes([Fetch.all.value])
        with self.lock:
            if self.topology.local:
                self.local.args.send(self.cache(data))
            if self.topology.relays:
                self.remote.args.send(data)
            results = [self.serializer.deserialize(self.local.result.recv()) for _ in range(len(self.topology.local))]
            results += list(itertools.chain(*[self.serializer.deserialize(self.remote.result.recv()) for _ in range(len(self.topology.relays))]))
        assert len(results) == len(self.actors), (len(results), len(self.actors))
        for result in results:
            if isinstance(result, Exception):
                raise result
        return results

    def __repr__(self):
        summary = [
            f"# {self.topology.host} local\n"
            f"{len(self.topology.local)} actors",
            f"# {self.topology.host} remote\n"
            f"{len(set(itertools.chain(*self.topology.nodes)))} actors",
            f"{len(self.topology.nodes)} nodes",
            f"{len(self.topology.relays)} relays",
        ]
        relay = [f'# {node.host} ({len(node.actors)} actors, {len(nodes)} nodes)\n{TABLE_HEADER + NL.join([f"| {node.host} | {len(node)} |" for node in nodes])}' for node, nodes in self.topology.relays.items()]
        return NLNL.join(summary + relay)
