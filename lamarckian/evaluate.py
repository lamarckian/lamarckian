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
import numbers
import argparse
import logging.config

import numpy as np
import torch
import torch.utils.tensorboard
import glom
import tqdm
import oyaml
import shutil
import ray

import lamarckian


def write_tb(items, root):
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root, exist_ok=True)
    with torch.utils.tensorboard.SummaryWriter(root) as writer:
        for item in items:
            for key, value in item['result'].items():
                writer.add_scalar(f"result{item['sample'] if 'sample' in item else ''}/{key}", value, int(item['cost']))


def main():
    args = make_args()
    config = {}
    for path in sum(args.config, []):
        config = lamarckian.util.config.read(path, config)
    if args.test:
        config = lamarckian.util.config.make_test(config)
    for cmd in sum(args.modify, []):
        lamarckian.util.config.modify(config, cmd)
    logging.config.dictConfig(config['logging'])
    logging.getLogger('filelock').setLevel(logging.ERROR)
    root = os.path.expanduser(os.path.expandvars(config['root']))
    with open(root.rstrip(os.sep) + '.yml', 'w') as f:
        f.write(oyaml.dump(config, allow_unicode=True))
    ray.init(**glom.glom(config, 'ray', default={}))
    Evaluator = lamarckian.evaluator.parse(*config['evaluator']['create'], config=config)
    with contextlib.closing(Evaluator(config=config)) as evaluator:
        items = []
        for path in lamarckian.util.file.load(root):
            logging.info(path)
            state = torch.load(path, map_location=lambda storage, loc: storage)
            if 'decision' in state:
                evaluator.set(state['decision'])
                results = state['results'] = evaluator.evaluate()
                state['result'] = evaluator.reduce(state['results'])
            elif 'population' in state:
                if args.all:
                    for index, individual in enumerate(tqdm.tqdm(state['population'])):
                        evaluator.set(individual['decision'])
                        individual['results'] = evaluator.evaluate()
                        individual['result'] = evaluator.reduce(individual['results'])
                    index = np.argmax([individual['result']['fitness'] for individual in state['population']])
                    elite = state['population'][index]
                    results = elite['results']
                elif args.index is not None:
                    individual = state['population'][args.index]
                    evaluator.set(individual['decision'])
                    results = individual['results'] = evaluator.evaluate()
                    individual['result'] = evaluator.reduce(individual['results'])
                else:
                    index = np.argmax([individual['result']['fitness'] for individual in state['population']])
                    elite = state['population'][index]
                    evaluator.set(elite['decision'])
                    results = elite['results'] = evaluator.evaluate()
                    elite['result'] = evaluator.reduce(elite['results'])
            if args.save:
                torch.save(state, path)
            result = evaluator.reduce(results)
            result = {key: value for key, value in result.items() if not key.startswith('_') and isinstance(value, numbers.Number)}
            items.insert(0, dict(
                cost=int(os.path.basename(os.path.splitext(path)[0])),
                result=result,
                sample=len(results),
            ))
            logging.info(result)
            with open(path + f".result{len(results)}", 'w') as f:
                f.write(str(result))
        write_tb(items, os.path.join(root, 'evaluate'))


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', action='append', default=[[os.path.dirname(os.path.abspath(lamarckian.__file__)) + '.yml']])
    parser.add_argument('-m', '--modify', nargs='+', action='append', default=[])
    parser.add_argument('-t', '--test', action='store_true')
    parser.add_argument('-a', '--all', action='store_true')
    parser.add_argument('-i', '--index', type=int)
    parser.add_argument('-s', '--save', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main()
