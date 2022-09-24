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
import functools
import contextlib
import inspect
import argparse
import logging.config

import numpy as np
import torch
import glom
import tqdm
import ray

import lamarckian


def surgery(blob, old, new, **kwargs):
    module = lamarckian.evaluator.parse(*new['cls'], **kwargs)
    model = module(**new['kwargs'], **kwargs)
    state_dict = model.state_dict()
    header_old, = old['kwargs']['inputs'][0]['header']
    header_new, = new['kwargs']['inputs'][0]['header']
    assert len(set(header_old)) == len(header_old), (len(header_old) - len(set(header_old)))
    assert len(set(header_new)) == len(header_new), (len(header_new) - len(set(header_new)))
    for i, (layer_old, layer_new) in enumerate(zip(blob, state_dict.values())):
        if layer_old.shape != layer_new.shape:
            logging.info(f"{layer_old.shape} -> {layer_new.shape}")
            neurons = {name: neuron for name, neuron in zip(header_old, layer_old.T)}
            assert len(neurons) == len(header_old)
            blob[i] = np.stack([neurons[name] if name in neurons else neuron.numpy() for name, neuron in zip(header_new, layer_new.T)], axis=-1)
            assert blob[i].shape == layer_new.shape, (blob[i].shape, layer_new.shape)
    return blob


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
    ray.init(**glom.glom(config, 'ray', default={}))
    Evaluator = lamarckian.evaluator.parse(*config['evaluator']['create'], config=config)
    with contextlib.closing(Evaluator(config=config)) as evaluator:
        old = state['encoding']['blob']['models'][evaluator.me]
        new = evaluator.describe()['blob']['models'][evaluator.me]
        if 'decision' in state:
            state['decision']['blob'] = surgery(state['decision']['blob'], old, new, config=config)
        elif 'population' in state:
            for individual in tqdm.tqdm(state['population']):
                individual['decision']['blob'] = surgery(individual['decision']['blob'], old, new, config=config)
    state.pop('opponents_train', None)
    for key in [key for key in state if key.startswith('lamarckian.rl.wrap.opponent.')]:
        del state[key]
    torch.save(state, path)
    logging.info(f"save {path}")


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', action='append', default=[[os.path.dirname(os.path.abspath(lamarckian.__file__)) + '.yml']])
    parser.add_argument('-m', '--modify', nargs='+', action='append', default=[])
    return parser.parse_args()


if __name__ == '__main__':
    main()