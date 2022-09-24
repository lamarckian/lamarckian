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
import sys
import contextlib
import traceback
import argparse
import logging
import logging.config

import numpy as np
import torch
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from PyQt5.QtWidgets import QApplication
import glom

import lamarckian


def create_ax(fig, dim):
    if dim == 3:
        ax = mpl_toolkits.mplot3d.Axes3D(fig)
        for name in ['x', 'y', 'z']:
            getattr(ax, f'set_{name}label')(name)
    else:
        ax = fig.gca()
    return ax


def draw(self, points, labels, color, **kwargs):
    dim = len(points[0])
    assert dim > 0
    ax = create_ax(self.fig, dim)
    cmap = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(np.min(color), np.max(color)), cmap=matplotlib.cm.jet)
    picker = glom.glom(kwargs['config'], 'gui.picker', default=3)
    marker = dict(default='o', remove='x')
    if dim > 3:
        marker = dict(default='-', remove='--')
        x = list(range(dim))
        artist = [ax.plot(x, objective, marker['default'], color=c, picker=picker)[0] for objective, c in zip(points, cmap.to_rgba(color))]
        ax.set_xticks(x, minor=False)
    elif dim == 1:
        artist = [ax.scatter(x, 0, color=c, picker=picker) for (x,), c in zip(points, cmap.to_rgba(color))]
    else:
        artist = [ax.scatter(*objective, color=c, picker=picker) for objective, c in zip(points, cmap.to_rgba(color))]
    self.fig.canvas.manager.set_window_title(f'({cmap.norm.vmin}, {cmap.norm.vmax})')
    return artist, marker


class Figure(object):
    def __init__(self, population, prefix=None, **kwargs):
        self.population = population
        self.prefix = prefix
        self.kwargs = kwargs
        self.fig = plt.figure()
        keys = glom.glom(kwargs['config'], 'ec.behavior')
        if keys:
            points = np.array([[glom.glom(individual['result'], key) for key in keys] for individual in population])
            labels = keys
        else:
            try:
                points = np.array([individual['result']['objective'] for individual in population])
                assert points.shape[1] > 0, points.shape
            except (KeyError, AssertionError):
                traceback.print_exc()
                points = np.array([[0, i] for i in range(len(population))])
            labels = [str(i) for i in range(len(points[0]))]
        try:
            color = np.array([individual['result']['fitness'] for individual in population])
        except KeyError:
            traceback.print_exc()
            color = np.array([0 for i in range(len(population))])
        self.artist, self.marker = draw(self, points, labels, color, **kwargs)
        self.fig.tight_layout()
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)

    def close(self):
        plt.close(self.fig)

    def on_pick(self, event):
        self.index = self.artist.index(event.artist)
        individual = self.population[self.index]
        logging.info(f"individual{self.index}: {individual.get('result', {})}")
        if event.mouseevent.button == 3:  # right click
            if self.prefix is not None:
                path = f"{self.prefix}_{self.index}.pth"
                torch.save(individual, path)
                logging.info('model saved into ' + path)
        if event.mouseevent.button == 2:  # middle click
            artist = self.artist[self.index]
            artist.set_marker(self.marker['remove'] if artist.get_marker() == self.marker['default'] else self.marker['default'])
            self.fig.canvas.draw()
            logging.warning('add' if artist.get_marker() == self.marker['default'] else 'remove' + f' individual{self.index}')

    def on_press(self, event):
        if event.key == 's':
            population = [individual for individual, artist in zip(self.population, self.artist) if artist.get_marker() == self.marker['default']]
            if len(population) < len(self.population):
                if self.prefix is not None:
                    path = f"{self.prefix}({len(population)}).pth"
                    torch.save(dict(population=population), path)
                    logging.info('filtered population saved into ' + path)


def main():
    args = make_args()
    config = {}
    for path in sum(args.config, []):
        config = lamarckian.util.config.read(path, config)
    for cmd in sum(args.modify, []):
        lamarckian.util.config.modify(config, cmd)
    logging.config.dictConfig(config['logging'])
    root = os.path.expanduser(os.path.expandvars(config['root']))
    logging.info(root)
    path = next(lamarckian.util.file.load(root))
    state = torch.load(path, map_location=lambda storage, loc: storage)
    with contextlib.closing(Figure(state['population'], root, config=config)):
        plt.show()


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', action='append', default=[[os.path.dirname(os.path.abspath(lamarckian.__file__)) + '.yml']])
    parser.add_argument('-m', '--modify', nargs='+', action='append', default=[])
    return parser.parse_args()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main()
