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

import contextlib
import pickle
import hashlib
import threading
import queue
import re

import numpy as np
import torch
import zmq

import lamarckian
from lamarckian.mdp import rollout
from . import broadcaster


class Truncator(lamarckian.rl.Truncator):
    def __init__(self, rl, step, cast=rollout.cast):
        super().__init__(rl, step, cast=cast)
        assert 0 < step < np.iinfo(np.int).max, step
        self.step = step

    def __next__(self):
        trajectory = []
        results = []
        while len(trajectory) < self.step:
            if self.done:
                self._reset()
            task = self.loop.create_task(rollout.get_trajectory(self.battle.controllers[0], self.agent, step=self.step - len(trajectory), cast=self.cast))
            task.add_done_callback(lamarckian.util.print_exc)
            lamarckian.mdp.util.wait(task, self.tasks, self.loop)
            trajectory1, exp = task.result()
            trajectory += trajectory1
            self.done = trajectory[-1]['done']
            if self.done:
                result = self.battle.controllers[0].get_result()
                if self.opponent:
                    result['digest_opponent_train'] = hashlib.md5(pickle.dumps(list(self.opponent.values()))).hexdigest()
                results.append(result)
                # self.loop.run_until_complete(asyncio.gather(*self.tasks))
        assert len(trajectory) == self.step, (len(trajectory), self.step)
        return trajectory, exp, results


class Gather(object):
    def __init__(self):
        self.cost = 0
        self.tensors = []
        self.results = []

    def __call__(self, cost, tensors, results):
        self.cost += cost
        self.tensors.append(tensors)
        self.results += results

    def get_tensors(self):
        assert self.tensors


class Prefetcher(object):
    def __init__(self, url, batch_size, dim=0, device=None, serializer='msgpack_lz4', capacity=1, **kwargs):
        assert batch_size > 0, batch_size
        self.batch_size = batch_size
        self.dim = dim
        if device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = device
        self.serializer = serializer
        self.deserialize = lamarckian.util.serialize.DESERIALIZE[serializer]
        self.cost = 0
        self.cache = []
        self.results = []
        self.iterations = []
        self.queue = queue.Queue(capacity)
        self.running = True
        self.thread = threading.Thread(target=self.run, args=(url,))
        self.thread.start()
        self.port = int(re.match(r'.+:(\d+)', url).group(1))

    def close(self):
        self.running = False
        context = zmq.Context()
        with contextlib.closing(context.socket(zmq.REQ)) as socket:
            socket.connect(f"tcp://127.0.0.1:{self.port}")
            socket.send(b'')
            socket.recv()
        self.thread.join()
        context.term()

    def gather(self, socket):
        size = 0
        while size < self.batch_size:
            data = socket.recv()
            socket.send(b'')
            if not data:
                raise KeyboardInterrupt()
            cost, tensors, results, iteration = self.deserialize(data)
            self.cost += cost
            tensors = lamarckian.rl.to_device(self.device, **tensors)
            self.cache.append(tensors.values())
            self.results += results
            self.iterations.append(iteration)
            size += tensors['reward'].shape[self.dim]
        cost, self.cost = self.cost, 0
        tensors, self.cache = {key: lamarckian.rl.cat(values, self.dim) for key, values in zip(tensors, zip(*self.cache))}, []
        results, self.results = self.results, []
        iterations, self.iterations = self.iterations, []
        return cost, tensors, results, iterations

    def run(self, url):
        context = zmq.Context()
        with contextlib.closing(context.socket(zmq.REP)) as socket:
            socket.bind(url)
            try:
                while self.running:
                    self.queue.put(self.gather(socket))
            except KeyboardInterrupt:
                pass
        context.term()

    def __call__(self, *args, **kwargs):
        return self.queue.get(*args, **kwargs)
