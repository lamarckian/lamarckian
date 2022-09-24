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
import contextlib
import threading
import argparse
import logging.config

import glom
import tqdm
import humanfriendly
import ray

import lamarckian


@ray.remote(num_cpus=1)
@lamarckian.util.rpc.wrap.all
@lamarckian.util.rpc.wrap.gather
class Actor(object):
    def __init__(self, size):
        self.data = os.urandom(size)

    def set(self, data):
        return len(data)

    def gather(self):
        return self.data


class Receiving(threading.Thread):
    def __init__(self, rpc_gather):
        super().__init__()
        self.rpc_gather = rpc_gather
        self.running = True
        self.start()

    def close(self):
        self.running = False
        self.join()

    def run(self):
        while self.running:
            self.rpc_gather()


def main():
    args = make_args()
    config = {}
    for path in sum(args.config, []):
        config = lamarckian.util.config.read(path, config)
    for cmd in sum(args.modify, []):
        lamarckian.util.config.modify(config, cmd)
    logging.config.dictConfig(config['logging'])
    logging.getLogger('filelock').setLevel(logging.ERROR)
    ray.init(**glom.glom(config, 'ray', default={}))
    data = os.urandom(humanfriendly.parse_size(args.send))
    actors = [Actor.remote(humanfriendly.parse_size(args.receive)) for _ in tqdm.trange(glom.glom(config, 'evaluator.parallel', default=int(ray.cluster_resources()['CPU'])), desc='create actors')]
    try:
        with tqdm.tqdm() as pbar, contextlib.closing(lamarckian.util.rpc.All(actors, config=config, progress='create RPC')) as rpc_all, contextlib.closing(lamarckian.util.rpc.Gather(rpc_all, config=config)) as rpc_gather, contextlib.closing(rpc_gather.gathering('gather')), contextlib.closing(Receiving(rpc_gather)), contextlib.closing(lamarckian.util.duration.Measure()) as duration:
            while True:
                rpc_all('set', data)
                pbar.update()
    except KeyboardInterrupt:
        logging.info(pbar.n / duration())


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', action='append', default=[[os.path.dirname(lamarckian.__file__) + '.yml']])
    parser.add_argument('-m', '--modify', nargs='+', action='append', default=[])
    parser.add_argument('-s', '--send', default='1.2m')
    parser.add_argument('-r', '--receive', default='0.2m')
    return parser.parse_args()


if __name__ == '__main__':
    main()
