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

import numpy as np
import glom
import humanfriendly

import lamarckian


def main():
    args = make_args()
    config = {}
    for path in sum(args.config, []):
        config = lamarckian.util.config.read(path, config)
    for cmd in sum(args.modify, []):
        lamarckian.util.config.modify(config, cmd)
    logging.config.dictConfig(config['logging'])
    serializer = lamarckian.util.rpc.util.Serializer(glom.glom(config, 'rpc.serializer'))
    size = humanfriendly.parse_size(args.size)
    data = serializer.serialize(os.urandom(size))
    try:
        with contextlib.closing(lamarckian.util.duration.Measure()) as duration:
            for n in range(args.total):
                serializer.deserialize(data)
    except KeyboardInterrupt:
        logging.info(n / duration.get())


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', action='append', default=[[os.path.dirname(lamarckian.__file__) + '.yml']])
    parser.add_argument('-m', '--modify', nargs='+', action='append', default=[])
    parser.add_argument('-s', '--size', default='1.2m')
    parser.add_argument('-t', '--total', type=int, default=np.iinfo(int).max)
    return parser.parse_args()


if __name__ == '__main__':
    main()
