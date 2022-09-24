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
import toolz
import tqdm
import oyaml
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
    population = torch.load(path, map_location=lambda storage, loc: storage)['population']
    ray.init(**glom.glom(config, 'ray', default={}))
    Evaluator = lamarckian.evaluator.parse(*config['evaluator']['create'], config=config)
    with contextlib.closing(Evaluator(config=config, root=root)) as evaluator:
        sample = glom.glom(config, 'sample.eval', default=len(evaluator))
        enemy, = evaluator.enemies
        items = [(i, j, seed) for i in range(len(population)) for j in range(len(population)) if j != i for seed in range(sample)]
        items = list(evaluator.rpc_any.map((('evaluate1', (seed, {enemy: population[j]['decision']['blob']}), dict(blob=population[i]['decision']['blob'], i=i, j=j)) for i, j, seed in tqdm.tqdm(items))))
        items = sorted(items, key=lambda item: (item[0], item[-1]['i'], item[-1]['j']))
        cache = {i: {j: np.mean([result['win'] - 0.5 for seed, cost, result in items]) for j, items in toolz.groupby(lambda item: item[-1]['j'], items).items()} for i, items in toolz.groupby(lambda item: item[-1]['i'], items).items()}
        payoff = np.full((len(population), len(population)), np.nan)
        for i in range(len(population)):
            payoff[i, i] = 0
        for i, combat in cache.items():
            for j, value in combat.items():
                payoff[i, j] = value
        torch.save(payoff, os.path.join(root, f"{args.name}.pth"))
        with open(os.path.join(root, f"{args.name}.yml"), 'w') as f:
            f.write(oyaml.dump(config, allow_unicode=True))
        payoff = np.full((len(population), len(population)), np.nan)
        for i in range(len(population)):
            for j in range(i):
                payoff[i, j] = cache[i][j] + cache[j][i]
        torch.save(payoff, os.path.join(root, f".{args.name}.pth"))


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', action='append', default=[[os.path.dirname(lamarckian.__file__) + '.yml']])
    parser.add_argument('-m', '--modify', nargs='+', action='append', default=[])
    parser.add_argument('-r', '--reverse', action='store_true')
    parser.add_argument('-n', '--name', default='payoff')
    return parser.parse_args()


if __name__ == '__main__':
    main()
