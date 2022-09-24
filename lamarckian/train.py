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
import types
import traceback
import argparse
import logging.config

import torch
import glom
import tqdm
import filelock
import shutil
import ray

import lamarckian


def tidy(root, args):
    os.makedirs(root, exist_ok=True)
    with filelock.FileLock(root + '.lock', 0):
        if args.delete:
            logging.warning('delete models in directory: ' + root)
            try:
                [os.remove(os.path.join(root, name)) for name in os.listdir(root) if name.endswith('.pth')]
            except FileNotFoundError:
                pass
        if args.DELETE:
            logging.warning('delete model directory: ' + root)
            shutil.rmtree(root, ignore_errors=True)
        os.makedirs(root, exist_ok=True)


def load(root):
    try:
        path = next(lamarckian.util.file.load(root))
        with contextlib.closing(lamarckian.util.DelayNewline()) as logger:
            logging.info(f'load {path} ... ')
            state = torch.load(path, map_location=lambda storage, loc: storage)
            logger('done')
        if 'decision' not in state:
            logging.warning('decision not found')
            state['decision'] = max(state['population'], key=lambda individual: individual['result']['fitness'])['decision']
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
    state = load(root)
    ray.init(**glom.glom(config, 'ray', default={}))
    Evaluator = lamarckian.evaluator.parse(*glom.glom(config, 'evaluator.create'), config=config)
    terminate = eval(f"lambda self, outcome: {glom.glom(config, 'evaluator.terminate', default='False')}")
    with contextlib.closing(Evaluator(state, config=config, root=root)) as evaluator, \
            filelock.FileLock(root + '.lock', 0), tqdm.tqdm(initial=evaluator.cost) as pbar, \
            torch.autograd.detect_anomaly() if args.debug else contextlib.closing(types.SimpleNamespace(close=lambda: None)):
        with contextlib.closing(evaluator.training()):
            try:
                while True:
                    cost = evaluator.cost
                    outcome = evaluator()
                    pbar.update(evaluator.cost - cost)
                    if terminate(evaluator, outcome):
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
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main()
