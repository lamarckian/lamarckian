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
import argparse
import logging.config

import numpy as np
import torch
import glom
import tqdm
import ray

import lamarckian


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
    path = next(lamarckian.util.file.load(root))
    logging.info(f'load {path}')
    state = torch.load(path, map_location=lambda storage, loc: storage)
    try:
        blob = state['decision']['blob']
    except KeyError:
        blob = max(state['population'], key=lambda individual: individual['result']['fitness'])['decision']['blob']
    ray.init(**glom.glom(config, 'ray', default={}))
    Evaluator = lamarckian.evaluator.parse(*config['evaluator']['create'], config=config)
    with contextlib.closing(Evaluator(config=config, root=root)) as evaluator:
        evaluator.set_blob(blob)
        enemy, = evaluator.enemies
        items = list(evaluator.rpc_any.map((('evaluate1', (seed, {enemy: blob}), {}) for seed in tqdm.trange(glom.glom(config, 'sample.eval', default=len(evaluator))))))
        items = sorted(items, key=lambda item: item[0])
        logging.info(np.mean([result['win'] - 0.5 for seed, cost, result in items]))


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', action='append', default=[[os.path.dirname(lamarckian.__file__) + '.yml']])
    parser.add_argument('-m', '--modify', nargs='+', action='append', default=[])
    return parser.parse_args()


if __name__ == '__main__':
    main()
