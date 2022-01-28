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

import sys
import os
import argparse
import logging.config

import numpy as np
import torch
import glom
import tqdm
import imageio
import imagecodecs
from PyQt5 import QtWidgets

import lamarckian


def main():
    args = make_args()
    config = {}
    for path in sum(args.config, []):
        config = lamarckian.util.config.read(path, config)
    for cmd in sum(args.modify, []):
        lamarckian.util.config.modify(config, cmd)
    logging.config.dictConfig(config['logging'])
    _ = QtWidgets.QApplication(sys.argv)
    root = os.path.expanduser(os.path.expandvars(config['root']))
    root = QtWidgets.QFileDialog.getExistingDirectory(caption='choose the rollout directory', directory=root)
    if root:
        mdp = torch.load(root + '.pth')
        mdp['me'] = mdp.get('me', 0)
        fps = glom.glom(config, 'mdp.fps')
        for index in sorted(list(map(int, filter(str.isdigit, {name.split('.')[0] for name in os.listdir(root)})))):
            prefix = os.path.join(root, str(index))
            logging.info(prefix)
            data = torch.load(f'{prefix}.pth')
            with imageio.get_writer(prefix + args.ext, fps=fps) as writer:
                for exp in tqdm.tqdm(data['trajectory']):
                    image = exp['state']['image']
                    if not isinstance(image, np.ndarray):
                        image = imagecodecs.jpeg_decode(image)
                    writer.append_data(image)


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', action='append', default=[[os.path.dirname(os.path.abspath(lamarckian.__file__)) + '.yml']])
    parser.add_argument('-m', '--modify', nargs='+', action='append', default=[])
    parser.add_argument('-e', '--ext', default='.mp4')
    return parser.parse_args()


if __name__ == '__main__':
    main()
