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
import traceback
import argparse
import logging.config

import numpy as np
import torch
import glom
import tqdm
import filelock
import ray

import lamarckian
from lamarckian.train import tidy


def load(root, **kwargs):
    try:
        path = next(lamarckian.util.file.load(root))
        logging.info(f'load {path}')
        state = torch.load(path, map_location=lambda storage, loc: storage)
        if 'population' not in state:
            logging.warning('population not in state')
            rs = np.random.RandomState(0)
            stddev = glom.glom(kwargs['config'], 'ec.mutation.blob.gaussian')
            blob = state['decision']['blob']
            population = [dict(decision=dict(blob=blob))]
            for _ in range(1, glom.glom(kwargs['config'], 'ec.population')):
                population.append(dict(decision=dict(blob=[value + rs.randn(*value.shape) * stddev for value in blob])))
            state['population'] = population
        return state
    except StopIteration:
        logging.warning(f'no model found in {root}')
        return {}


def main():
    args = make_args()
    config = {}
    for path in sum(args.config, []):
        config = lamarckian.util.config.read(path, config)
    for cmd in sum(args.modify, []):
        lamarckian.util.config.modify(config, cmd)
    logging.config.dictConfig(config['logging'])
    logging.getLogger('filelock').setLevel(logging.ERROR)
    root = os.path.expanduser(os.path.expandvars(config['root']))
    tidy(root, args)
    state = load(root, config=config)
    ray.init(**glom.glom(config, 'ray', default={}))
    EC = lamarckian.evaluator.parse(*config['ec']['create'], config=config)
    terminate = eval(f"lambda self, outcome: {glom.glom(config, 'ec.terminate', default='False')}")
    with contextlib.closing(EC(state, config=config, root=root)) as ec, \
            filelock.FileLock(root + '.lock', 0), tqdm.tqdm(initial=ec.cost) as pbar:
        with contextlib.closing(ec.evolving()):
            try:
                while True:
                    cost = ec.cost
                    outcome = ec()
                    assert cost < ec.cost
                    pbar.update(ec.cost - cost)
                    if terminate(ec, outcome):
                        break
            except KeyboardInterrupt:
                logging.warning('keyboard interrupted')
            except:
                traceback.print_exc()
                raise
            finally:
                logging.info(lamarckian.util.duration.stats)


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', action='append', default=[[os.path.dirname(os.path.abspath(lamarckian.__file__)) + '.yml']])
    parser.add_argument('-m', '--modify', nargs='+', action='append', default=[])
    parser.add_argument('-d', '--delete', action='store_true')
    parser.add_argument('-D', '--DELETE', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main()
