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

import glom
import toolz
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
    ray.init(**glom.glom(config, 'ray', default={}))
    actors = ray.actors()
    for ip, group in toolz.groupby(lambda item: item['Address']['IPAddress'], actors.values()).items():
        print(f"{ip} ({len(group)})")
        for item in sorted(group, key=lambda item: item['Name']):
            print(item['Name'])
        print()


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', action='append', default=[[os.path.dirname(os.path.abspath(lamarckian.__file__)) + '.yml']])
    parser.add_argument('-m', '--modify', nargs='+', action='append', default=[])
    return parser.parse_args()


if __name__ == '__main__':
    main()
