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
import copy
import contextlib
import argparse
import logging.config

import numpy as np
import torch
import glom
import matplotlib.pyplot as plt
import tqdm
import oyaml
import imageio
import imagecodecs
import ray

import lamarckian


def rollout(evaluator, root, **kwargs):
    os.makedirs(root, exist_ok=True)
    torch.save(evaluator.__getstate__(), root + '.mdp.pth')
    try:
        sample = glom.glom(kwargs['config'], 'sample.eval', default=len(evaluator))
        results = []
        for seed, cost, trajectory, exp, result in tqdm.tqdm(evaluator.iterate_trajectory(sample, message=True), total=sample):
            results.append(result)
            logging.info(result)
            torch.save(dict(trajectory=trajectory, exp=exp, result=result), os.path.join(root, f"{seed}.pth"))
            if kwargs['video']:
                fps = glom.glom(kwargs['config'], 'mdp.fps')
                path = os.path.join(root, f"{seed}.mp4")
                with imageio.get_writer(path, fps=fps) as writer:
                    for exp in tqdm.tqdm(trajectory, desc=f'save {path}'):
                        image = exp['state']['image']
                        if not isinstance(image, np.ndarray):
                            image = imagecodecs.jpeg_decode(image)
                        writer.append_data(image)
        logging.info(evaluator.reduce(results))
    except KeyboardInterrupt:
        logging.warning('keyboard interrupt')


def load(path, **kwargs):
    state = torch.load(path, map_location=lambda storage, loc: storage)
    if 'population' in state:
        from lamarckian.select import Figure
        population = state['population']
        try:
            with contextlib.closing(Figure(population, os.path.splitext(path)[0], **kwargs)) as figure:
                plt.show()
            assert hasattr(figure, 'index')
            state['decision'] = population[figure.index]['decision']
        except AssertionError:
            state['decision'] = max(population, key=lambda individual: individual['result']['fitness'])['decision']
    return state


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
    path = next(lamarckian.util.file.load(root))
    logging.info(path)
    with open(path + '.yml', 'w') as f:
        f.write(oyaml.dump(config, allow_unicode=True))
    state = load(path, config=config)
    prefix = os.path.join(os.path.dirname(path), 'rollout', os.path.basename(os.path.splitext(path)[0]))
    ray.init(**glom.glom(config, 'ray', default={}))
    Evaluator = lamarckian.evaluator.parse(*config['evaluator']['create'], config=config)
    with contextlib.closing(Evaluator(state, config=config)) as evaluator:
        rollout(evaluator, prefix, config=config, video=args.video)


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', action='append', default=[[os.path.dirname(os.path.abspath(lamarckian.__file__)) + '.yml']])
    parser.add_argument('-m', '--modify', nargs='+', action='append', default=[])
    parser.add_argument('-t', '--test', action='store_true')
    parser.add_argument('-v', '--video', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main()
