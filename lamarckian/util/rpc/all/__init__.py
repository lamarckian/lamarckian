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
import itertools
import contextlib
import types
import math
import struct
import binascii
import threading
import time
from timeit import default_timer as timer

import zmq
import glom
import toolz
import humanfriendly
import tqdm
import filelock
import port_for
import ray.exceptions

import lamarckian
from .. import util


def get_hosts(self, **kwargs):
    try:
        tasks = [actor.setup_rpc_all.remote() for actor in self.actors]
        desc = kwargs.get('progress', None)
        if desc is not None:
            lamarckian.util.remote.progress(tasks, f"{desc} test", self.ray)
        return self.ray.get(tasks)
    except ray.exceptions.RayActorError:
        raise lamarckian.util.remote.RayActorError(f"{', '.join(util.detect_died_actors(self.actors, lambda actor: actor.setup_rpc_all.remote()))} died")


def ensure(duration):
    start = timer()

    def close():
        elapsed = timer() - start
        if elapsed < duration:
            time.sleep(duration - elapsed)
    return contextlib.closing(types.SimpleNamespace(close=close))


class Flat(object):
    def __init__(self, actors, **kwargs):
        assert actors
        self.actors = actors
        self.kwargs = kwargs
        self.ray = kwargs.get('ray', ray)
        self.host = ray.util.get_node_ip_address()
        serializer = glom.glom(kwargs['config'], 'rpc.serializer')
        self.serializer = lamarckian.util.serialize.Serializer(kwargs.get('serializer', serializer))
        self.setup(*self.bind())
        time.sleep(glom.glom(kwargs['config'], 'rpc.sleep', default=0))  # zmq slow joiner
        self.wait()
        self.lock = threading.Lock()

    def bind(self):
        self.context = zmq.Context(len(self.actors))
        self.socket = types.SimpleNamespace(param=self.context.socket(zmq.PUB), result=self.context.socket(zmq.PULL))
        ports = glom.glom(self.kwargs['config'], 'rpc.ports', default=None)
        with filelock.FileLock('/tmp/lamarckian.port'):
            param = port_for.select_random(ports)
            self.socket.param.bind(f"tcp://*:{param}")
            result = port_for.select_random(ports)
            self.socket.result.bind(f"tcp://*:{result}")
        return param, result

    def setup(self, param, result):
        self.ray.get([actor.setup_rpc_all.remote(
            'execute',
            f"tcp://{self.host}:{param}", f"tcp://{self.host}:{result}",
            **self.kwargs,
        ) for actor in self.actors])

    def wait(self):
        for _ in self.actors:
            self.socket.result.recv()

    def close(self):
        with self.lock:
            self.socket.param.send(b'')
            for _ in self.actors:
                self.socket.result.recv()
        self.socket.param.close()
        self.socket.result.close()
        self.context.term()

    def __len__(self):
        return len(self.actors)

    def __call__(self, _, *args, **kwargs):
        data = self.serializer.serialize((_, args, kwargs))
        with self.lock:
            self.socket.param.send(data + bytes([util.Fetch.call.value]))
            datum = [self.socket.result.recv() for _ in self.actors]
        for data in datum:
            if data[-1]:
                raise self.serializer.deserialize(data[:-1])

    def fetch_any(self, _, *args, **kwargs):
        data = self.serializer.serialize((_, args, kwargs))
        with self.lock:
            self.socket.param.send(data + bytes([util.Fetch.any.value]))
            datum = [self.socket.result.recv() for _ in self.actors]
        for data in datum:
            if data[-1]:
                raise self.serializer.deserialize(data[:-1])
        return self.serializer.deserialize(data[:-1])

    def fetch_all(self, _, *args, **kwargs):
        data = self.serializer.serialize((_, args, kwargs))
        with self.lock:
            self.socket.param.send(data + bytes([util.Fetch.all.value]))
            datum = [self.socket.result.recv() for _ in self.actors]
            results = [self.serializer.deserialize(data) for data in datum]
        for result in results:
            if isinstance(result, Exception):
                raise result
        return results


class Hybrid(object):
    def __init__(self, actors, profile=False, **kwargs):
        assert actors
        self.actors = actors
        self.kwargs = kwargs
        self.ray = kwargs.get('ray', ray)
        self.host = ray.util.get_node_ip_address()
        serializer = glom.glom(kwargs['config'], 'rpc.serializer')
        self.serializer = lamarckian.util.serialize.Serializer(kwargs.get('serializer', serializer))
        hosts = get_hosts(self, **kwargs)
        self.topology(hosts)
        shards = min(len(nodes) for nodes in self.branches)
        self.shards = min(glom.glom(self.kwargs['config'], 'rpc.all.shards', default=shards), shards)
        self.setup(*self.bind())
        time.sleep(glom.glom(kwargs['config'], 'rpc.sleep', default=0))  # zmq slow joiner
        self.wait()
        self.lock = threading.Lock()
        if profile:
            self.profile = {}

    def topology(self, hosts):
        nodes = [util.Node(host, [actor for _, actor in actors]) for host, actors in toolz.groupby(lambda item: item[0], zip(hosts, self.actors)).items()]
        fake = glom.glom(self.kwargs['config'], 'rpc.all.fake', default=0)
        if fake:
            actors = self.actors
            for _ in tqdm.trange(fake, desc='create fake nodes'):
                nodes.append(util.Node(binascii.b2a_hex(os.urandom(util.ID_LENGTH)).decode(), list(nodes[0])))
                self.actors = self.actors + actors
        branches = math.floor(math.sqrt(len(nodes)))
        branches = min(glom.glom(self.kwargs['config'], 'rpc.all.branches', default=branches), branches)
        self.branches = list(toolz.partition_all(math.ceil(len(nodes) / branches), nodes))

    def bind(self):
        self.context = zmq.Context(len(self.branches))
        self.sockets = [types.SimpleNamespace(param=self.context.socket(zmq.PUB), result=self.context.socket(zmq.PULL)) for _ in range(self.shards)]
        ports = []
        _ = glom.glom(self.kwargs['config'], 'rpc.ports', default=None)
        with filelock.FileLock('/tmp/lamarckian.port'):
            for socket in self.sockets:
                socket.param.setsockopt(zmq.SNDHWM, 0)
                socket.result.setsockopt(zmq.RCVHWM, 0)
                socket.param.setsockopt(zmq.XPUB_NODROP, 1)
                param = port_for.select_random(_)
                socket.param.bind(f"tcp://*:{param}")
                result = port_for.select_random(_)
                socket.result.bind(f"tcp://*:{result}")
                ports.append((param, result))
        return ports

    def setup(self, *args):
        shm = humanfriendly.parse_size(str(glom.glom(self.kwargs['config'], 'rpc.shm', default=0)))
        for b, nodes in enumerate(self.branches):
            routers = self.ray.get([nodes[index][0].setup_rpc_all.remote(
                'router',
                f"tcp://{self.host}:{param}", f"tcp://{self.host}:{result}",
                nodes,
                **{**self.kwargs, **dict(index=index, suffix=f".branch{b}")},
            ) for index, (param, result) in enumerate(args)])
            assemblers = self.ray.get([node[0].setup_rpc_all.remote(
                'assembler',
                routers, node,
                **{**self.kwargs, **dict(index=index, shm=shm, suffix=f".branch{b}")},
            ) for index, node in enumerate(nodes)])
            self.ray.get([actor.setup_rpc_all.remote(
                'execute',
                param, result,
                **{**self.kwargs, **dict(shm=shm, shm_name=shm_name)},
            ) for node, (param, result, shm_name) in zip(nodes, assemblers) for actor in node])

    def wait(self):
        for _ in self.branches:
            for socket in self.sockets:
                socket.result.recv()

    def close(self):
        with self.lock:
            for socket in self.sockets:
                socket.param.send(b'')
            for _ in self.branches:
                for socket in self.sockets:
                    socket.result.recv()
        for socket in self.sockets:
            socket.param.close()
            socket.result.close()
        self.context.term()

    def __len__(self):
        return len(self.actors)

    def __call__(self, _, *args, **kwargs):
        time = dict(_=timer())
        data = self.serializer.serialize((_, args, kwargs))
        time['serialize'] = timer()
        with self.lock:
            fetch = bytes([util.Fetch.call.value])
            size = math.ceil(len(data) / len(self.sockets))
            for i, socket in enumerate(self.sockets):
                chunk = data[i * size:(i + 1) * size]
                socket.param.send(chunk + fetch)
            datum = [socket.result.recv() for socket in self.sockets for _ in self.branches]
            datum = [data for data in datum if data]
        for data in datum:
            if data[-1]:
                raise self.serializer.deserialize(data[:-1])
        if hasattr(self, 'profile'):
            assembler, execute, _ = struct.unpack('<dd?', data[-(8 * 2 + 1):])
            self.profile = dict(
                serialize=time['serialize'] - time['_'],
                router=timer() - time['serialize'],
                assembler=assembler,
                execute=execute,
            )

    def fetch_any(self, _, *args, **kwargs):
        data = self.serializer.serialize((_, args, kwargs))
        with self.lock:
            fetch = bytes([util.Fetch.any.value])
            size = math.ceil(len(data) / len(self.sockets))
            for i, socket in enumerate(self.sockets):
                chunk = data[i * size:(i + 1) * size]
                socket.param.send(chunk + fetch)
            datum = [socket.result.recv() for socket in self.sockets for _ in self.branches]
            datum = [data for data in datum if data]
        for data in datum:
            if data[-1]:
                raise self.serializer.deserialize(data[:-1])
        return self.serializer.deserialize(data[:-1])

    def fetch_all(self, _, *args, **kwargs):
        data = self.serializer.serialize((_, args, kwargs))
        with self.lock:
            fetch = bytes([util.Fetch.all.value])
            size = math.ceil(len(data) / len(self.sockets))
            for i, socket in enumerate(self.sockets):
                chunk = data[i * size:(i + 1) * size]
                socket.param.send(chunk + fetch)
            datum = [socket.result.recv() for socket in self.sockets for _ in self.branches]
            results = list(itertools.chain(*[self.serializer.deserialize(data) for data in datum]))
        for result in results:
            if isinstance(result, Exception):
                raise result
        return results

    def __repr__(self):
        msg = ''
        for b, nodes in enumerate(self.branches):
            msg += f"# branch {b} ({len(nodes)} nodes)\n"
            msg += '| node | actors | router |\n'
            msg += '| :-- | :-- | :-- |\n'
            for i, node in enumerate(nodes):
                msg += f"| {node.host} | {len(node)} | {'*' if i < self.shards else ''} |\n"
            msg += '\n'
        msg += f"{sum(len(nodes) for nodes in self.branches)} nodes, {len(self.actors)} actors"
        return msg


class Tree(object):
    def __init__(self, actors, profile=False, **kwargs):
        assert actors
        self.actors = actors
        self.kwargs = kwargs
        self.ray = kwargs.get('ray', ray)
        self.host = ray.util.get_node_ip_address()
        serializer = glom.glom(kwargs['config'], 'rpc.serializer')
        self.serializer = lamarckian.util.serialize.Serializer(kwargs.get('serializer', serializer))
        hosts = get_hosts(self, **kwargs)
        self.topology(hosts)
        self.setup(*self.bind())
        time.sleep(glom.glom(kwargs['config'], 'rpc.sleep', default=0))  # zmq slow joiner
        self.wait()
        self.lock = threading.Lock()
        if profile:
            self.profile = {}

    def topology(self, hosts):
        nodes = [util.Node(host, [actor for _, actor in actors]) for host, actors in toolz.groupby(lambda item: item[0], zip(hosts, self.actors)).items()]
        fake = glom.glom(self.kwargs['config'], 'rpc.all.fake', default=0)
        if fake:
            actors = self.actors
            for _ in tqdm.trange(fake, desc='create fake nodes'):
                nodes.append(util.Node(binascii.b2a_hex(os.urandom(util.ID_LENGTH)).decode(), list(nodes[0])))
                self.actors = self.actors + actors
        branches = math.floor(math.sqrt(len(nodes)))
        branches = min(glom.glom(self.kwargs['config'], 'rpc.all.branches', default=branches), branches)
        self.branches = list(toolz.partition_all(math.ceil(len(nodes) / branches), nodes))

    def bind(self):
        self.context = zmq.Context(len(self.branches))
        self.socket = types.SimpleNamespace(param=self.context.socket(zmq.PUB), result=self.context.socket(zmq.PULL))
        _ = glom.glom(self.kwargs['config'], 'rpc.ports', default=None)
        with filelock.FileLock('/tmp/lamarckian.port'):
            param = port_for.select_random(_)
            self.socket.param.bind(f"tcp://*:{param}")
            result = port_for.select_random(_)
            self.socket.result.bind(f"tcp://*:{result}")
        return param, result

    def setup(self, param, result):
        shm = humanfriendly.parse_size(str(glom.glom(self.kwargs['config'], 'rpc.shm', default=0)))
        routers = self.ray.get([nodes[0][0].setup_rpc_all.remote(
            'router',
            f"tcp://{self.host}:{param}", f"tcp://{self.host}:{result}",
            nodes,
            **{**self.kwargs, **dict(index=index)},
        ) for index, nodes in enumerate(self.branches)])
        for b, (nodes, (param, result)) in enumerate(zip(self.branches, routers)):
            leafs = self.ray.get([node[0].setup_rpc_all.remote(
                'leaf',
                param, result,
                node,
                **{**self.kwargs, **dict(shm=shm, index=index, suffix=f".branch{b}")},
            ) for index, node in enumerate(nodes)])
            self.ray.get([actor.setup_rpc_all.remote(
                'execute',
                param, result,
                **{**self.kwargs, **dict(shm=shm, shm_name=shm_name)},
            ) for node, (param, result, shm_name) in zip(nodes, leafs) for actor in node])

    def wait(self):
        for _ in self.branches:
            self.socket.result.recv()

    def close(self):
        with self.lock:
            self.socket.param.send(b'')
            for _ in self.branches:
                self.socket.result.recv()
        self.socket.param.close()
        self.socket.result.close()
        self.context.term()

    def __len__(self):
        return len(self.actors)

    def __call__(self, _, *args, **kwargs):
        time = timer()
        data = self.serializer.serialize((_, args, kwargs))
        with self.lock:
            self.socket.param.send(data + bytes([util.Fetch.call.value]))
            datum = [self.socket.result.recv() for _ in self.branches]
        for data in datum:
            if data[-1]:
                raise self.serializer.deserialize(data[:-1])
        if hasattr(self, 'profile'):
            leaf, execute, _ = struct.unpack('<dd?', data[-(8 * 2 + 1):])
            self.profile = dict(
                total=timer() - time,
                leaf=leaf,
                execute=execute,
            )

    def fetch_any(self, _, *args, **kwargs):
        data = self.serializer.serialize((_, args, kwargs))
        with self.lock:
            self.socket.param.send(data + bytes([util.Fetch.any.value]))
            datum = [self.socket.result.recv() for _ in self.branches]
        for data in datum:
            if data[-1]:
                raise self.serializer.deserialize(data[:-1])
        return self.serializer.deserialize(data[:-1])

    def fetch_all(self, _, *args, **kwargs):
        data = self.serializer.serialize((_, args, kwargs))
        with self.lock:
            self.socket.param.send(data + bytes([util.Fetch.all.value]))
            datum = [self.socket.result.recv() for _ in self.branches]
            results = list(itertools.chain(*[self.serializer.deserialize(data) for data in datum]))
        for result in results:
            if isinstance(result, Exception):
                raise result
        return results

    def __repr__(self):
        msg = ''
        for b, nodes in enumerate(self.branches):
            msg += f"# branch {b} ({len(nodes)} nodes)\n"
            msg += '| node | actors | router |\n'
            msg += '| :-- | :-- | :-- |\n'
            for i, node in enumerate(nodes):
                msg += f"| {node.host} | {len(node)} | {'*' if i == 0 else ''} |\n"
            msg += '\n'
        msg += f"{sum(len(nodes) for nodes in self.branches)} nodes, {len(self.actors)} actors"
        return msg
