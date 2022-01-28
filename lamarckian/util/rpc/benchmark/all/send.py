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
import contextlib
import argparse
import logging.config

import glom
import tqdm
import humanfriendly
import ray

import lamarckian


@ray.remote(num_cpus=1)
@lamarckian.util.rpc.wrap.all
class Actor(object):
    def set(self, data):
        return len(data)


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
    size = humanfriendly.parse_size(args.size)
    data = os.urandom(size)
    actors = [Actor.remote() for _ in tqdm.trange(glom.glom(config, 'evaluator.parallel', default=int(ray.cluster_resources()['CPU'])), desc='create actors')]
    try:
        with tqdm.tqdm() as pbar, contextlib.closing(lamarckian.util.rpc.All(actors, config=config, progress='create RPC')) as rpc, contextlib.closing(lamarckian.util.duration.Measure()) as duration:
            while True:
                rpc('set', data)
                pbar.update()
    except KeyboardInterrupt:
        logging.info(pbar.n / duration.get())


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', action='append', default=[[os.path.dirname(lamarckian.__file__) + '.yml']])
    parser.add_argument('-m', '--modify', nargs='+', action='append', default=[])
    parser.add_argument('-s', '--size', default='1.2m')
    return parser.parse_args()


if __name__ == '__main__':
    main()
