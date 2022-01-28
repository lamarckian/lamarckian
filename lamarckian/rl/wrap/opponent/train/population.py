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
import argparse
import logging.config
import itertools

import numpy as np
import torch

import lamarckian


def main():
    args = make_args()
    config = {}
    for path in sum(args.config, []):
        config = lamarckian.util.config.read(path, config)
    for cmd in sum(args.modify, []):
        lamarckian.util.config.modify(config, cmd)
    logging.config.dictConfig(config['logging'])
    root = os.path.expanduser(os.path.expandvars(config['root']))
    path = next(lamarckian.util.file.load(root))
    logging.info(path)
    state = torch.load(path, map_location=lambda storage, loc: storage)
    opponents_train = state['opponents_train']
    state['population'] = [dict(decision=dict(blob=next(iter(blobs.values())))) for blobs in itertools.islice(reversed(opponents_train), args.limit)]
    torch.save(state, path)


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', action='append', default=[[os.path.dirname(os.path.abspath(lamarckian.__file__)) + '.yml']])
    parser.add_argument('-m', '--modify', nargs='+', action='append', default=[])
    parser.add_argument('-l', '--limit', type=int, default=np.iinfo(np.int).max)
    return parser.parse_args()


if __name__ == '__main__':
    main()
