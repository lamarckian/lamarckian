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
import itertools
import logging

import torch
import glom

from . import train, eval


def file(section, method):
    def load(path, spec):
        state = torch.load(path)
        try:
            blobs = [glom.glom(individual, spec) for individual in state['population']]
            try:
                _, indexes = os.path.basename(os.path.splitext(path)[0]).split(':')
                indexes = list(map(int, indexes.split(',')))
                blobs = [blobs[index] for index in indexes]
            except ValueError:
                pass
        except KeyError:
            blobs = [glom.glom(state, spec)]
        return blobs

    def decorate(rl):
        class RL(rl):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                root = os.path.expanduser(os.path.expandvars(glom.glom(kwargs['config'], f'{section}.root')))
                spec = glom.glom(kwargs['config'], f'{section}.spec', default='decision.blob')
                spec = [int(key) if key.lstrip('-+').isdigit() else key for key in spec.split('.')]
                spec = functools.reduce(lambda T, key: T[key], [glom.T] + spec)
                paths = sorted([os.path.join(root, filename) for filename in os.listdir(root) if filename.endswith('.pth')])
                _opponents = {enemy: list(itertools.chain(*(load(path, spec) for path in paths))) for enemy in self.enemies}
                opponents = [{key: blob for key, blob in zip(_opponents, opponent)} for opponent in zip(*_opponents.values())]
                assert opponents
                getattr(self, f'set_{method}')(opponents)
                logging.info(f'load {len(opponents)} {method} in {root}')
        return RL
    return decorate
