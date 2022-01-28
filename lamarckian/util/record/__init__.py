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
import tarfile
import logging

import numpy as np
import torch
import scipy.spatial.distance
import matplotlib.pyplot as plt
import inflection

from . import plot


class Reset(object):
    def __init__(self, cost):
        self.cost = cost

    def __call__(self, recorder):
        root_log = recorder.writer.log_dir
        recorder.writer.close()
        paths = [os.path.join(root_log, filename) for filename in os.listdir(root_log) if filename.startswith('events.out.tfevents.')]
        path = os.path.join(root_log, f'{self.cost}.tar.gz')
        with tarfile.open(path, 'w:gz') as tar:
            for p in paths:
                tar.add(p, arcname=os.path.basename(p))
        logging.info(path)
        for p in paths:
            os.remove(p)
        recorder.writer = type(recorder.writer)(root_log)


class Scalar(object):
    def __init__(self, cost, **kwargs):
        self.cost = cost
        self.kwargs = kwargs

    def __call__(self, recorder):
        for key, value in self.kwargs.items():
            recorder.writer.add_scalar(key, value, self.cost)


class Scalars(object):
    def __init__(self, cost, tag, **kwargs):
        self.cost = cost
        self.tag = tag
        self.kwargs = kwargs

    def __call__(self, recorder):
        recorder.writer.add_scalars(self.tag, self.kwargs, self.cost)


class Vector(object):
    def __init__(self, cost, **kwargs):
        self.cost = cost
        self.kwargs = kwargs

    def __call__(self, recorder):
        for key, vector in self.kwargs.items():
            for i, value in enumerate(vector):
                recorder.writer.add_scalar(f'{key}{i}', value, self.cost)


class Flat(object):
    def __init__(self, cost, **kwargs):
        self.cost = cost
        self.kwargs = kwargs

    def __call__(self, recorder):
        for key, points in self.kwargs.items():
            if points.size:
                matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(points))
                recorder.writer.add_scalar('/'.join([key, 'dist_mean']), np.mean(matrix), self.cost)
                recorder.writer.add_scalar('/'.join([key, 'dist_max']), np.max(matrix), self.cost)


class Histogram(object):
    def __init__(self, cost, **kwargs):
        self.cost = cost
        self.kwargs = kwargs

    def __call__(self, recorder):
        for key, var in self.kwargs.items():
            recorder.writer.add_histogram(key, var, self.cost)


class Embedding(object):
    def __init__(self, cost, tag, **kwargs):
        self.cost = cost
        self.tag = tag
        self.kwargs = kwargs

    def __call__(self, recorder):
        try:
            cmap = self.kwargs['cmap']
        except KeyError:
            cmap = plt.cm.jet
        try:
            repeat = self.kwargs['repeat']
        except KeyError:
            repeat = 3
        data, label = (self.kwargs[key] for key in 'data, label'.split(', '))
        if 'fitness' in self.kwargs:
            fitness = self.kwargs['fitness']
            a, b = min(fitness), max(fitness)
            r = b - a
            if r > 0:
                colors = [cmap(int((s - a) / r * cmap.N)) for s in fitness]
                images = torch.from_numpy(np.reshape([color[:3] for color in colors], [-1, 3, 1, 1]).repeat(repeat, 2).repeat(repeat, 3))
                images = (images * 255).float()
            else:
                images = None
        else:
            images = None
        recorder.writer.add_embedding(data, label, label_img=images, global_step=self.cost, tag=self.tag)


class Text(object):
    def __init__(self, cost, **kwargs):
        self.cost = cost
        self.kwargs = kwargs

    def __call__(self, recorder):
        for key, value in self.kwargs.items():
            recorder.writer.add_text(key, value, self.cost)


class Distribution(object):
    def __init__(self, cost, population, **kwargs):
        self.cost = cost
        self.population = population
        self.kwargs = kwargs

    def __call__(self, recorder):
        name = inflection.underscore(type(self).__name__)
        stamp_index = {individual['stamp']: i for i, individual in enumerate(self.population)}
        for tag, ids in self.kwargs.items():
            for i, id in enumerate(ids):
                individual = self.population[stamp_index[id]]
                if 'tags' in individual:
                    individual['tags'][tag] = i
                else:
                    individual['tags'] = {tag: i}
        root = os.path.join(recorder.root_model, name)
        os.makedirs(root, exist_ok=True)
        torch.save(self.population, os.path.join(root, f'{self.cost}.pth'))
