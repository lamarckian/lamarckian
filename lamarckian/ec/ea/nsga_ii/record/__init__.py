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
import itertools
import collections

import numpy as np
import torch
import glom
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt

import lamarckian


def get_objective(cost, population, tag='objective/population', **kwargs):
    objective = np.array([individual['result']['objective'] for individual in population])
    color = np.array([glom.glom(individual, 'result.fitness', default=0) for individual in population])
    return lamarckian.util.record.plot.Scatter(tag, cost, objective, c=color)


class Offspring(object):
    def __init__(self, tag, cost, offspring, **kwargs):
        self.tag = tag
        self.cost = cost
        self.offspring = [dict(origin=individual['origin'], result=dict(fitness=individual['result']['fitness'], objective=individual['result']['objective'])) for individual in offspring]
        self.kwargs = kwargs
        self.dim, = {len(individual['result']['objective']) for individual in offspring}
        assert 1 < self.dim < 4, self.dim

    def __call__(self, recorder):
        fig = plt.figure()
        ax = lamarckian.util.mpl.gca(fig, self.dim)
        count = collections.OrderedDict(degenerate=0, between=0, improve=0)
        for individual, prop in zip(self.offspring, itertools.cycle(plt.rcParams['axes.prop_cycle'])):
            fitness = np.array([result['fitness'] for result in individual['origin'].values()])
            if individual['result']['fitness'] > fitness.max():
                count['improve'] += 1
                linestyle = '-'
            elif individual['result']['fitness'] < fitness.min():
                count['degenerate'] += 1
                linestyle = ':'
            else:
                count['between'] += 1
                linestyle = '--'
            for result in individual['origin'].values():
                ax.arrow(*result['objective'], *(np.array(individual['result']['objective']) - result['objective']), color=prop['color'], linestyle=linestyle, alpha=0.3)
        ax.set_title(', '.join(f'{key}={value}' for key, value in count.items()))
        ancestor = list(itertools.chain(*[individual['origin'].values() for individual in self.offspring]))
        color = np.array([result['fitness'] for result in ancestor] + [individual['result']['fitness'] for individual in self.offspring])
        cmap = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(np.min(color), np.max(color)), cmap=matplotlib.cm.jet)
        # ax.scatter(*zip(*[result['objective'] for result in ancestor]), c=[cmap.to_rgba(result['fitness']) for result in ancestor], marker='o')
        ax.scatter(*zip(*[individual['result']['objective'] for individual in self.offspring]), c=[cmap.to_rgba(individual['result']['fitness']) for individual in self.offspring], marker='.')
        fig.canvas.draw()
        image = lamarckian.util.mpl.to_image(fig)
        prefix = os.path.join(self.kwargs['root'], self.tag, str(self.cost))
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        fig.savefig(prefix + '.svg')
        torch.save(self.offspring, prefix + '.pth')
        plt.close(fig)
        recorder.writer.add_image(self.tag, np.transpose(image, [2, 0, 1]), self.cost)
        for key, value in count.items():
            recorder.writer.add_scalar(f'{self.tag}/{key}', value, self.cost)


class Offspring_(Offspring):
    def __init__(self, tag, cost, offspring, **kwargs):
        if '_result' in offspring[0]:
            super().__init__(tag, cost, [dict(origin=individual['origin'], result=individual['_result']) for individual in offspring], **kwargs)

    def __call__(self, *args, **kwargs):
        if hasattr(self, 'offspring'):
            return super().__call__(*args, **kwargs)


class Train(object):
    def __init__(self, tag, cost, offspring, **kwargs):
        self.tag = tag
        self.cost = cost
        try:
            self.offspring = [dict(_result=dict(fitness=individual['_result']['fitness'], objective=individual['_result']['objective']), result=dict(fitness=individual['result']['fitness'], objective=individual['result']['objective'])) for individual in offspring]
        except KeyError:
            pass
        self.kwargs = kwargs
        self.dim, = {len(individual['result']['objective']) for individual in offspring}
        assert 1 < self.dim < 4, self.dim

    def __call__(self, recorder):
        if hasattr(self, 'offspring'):
            fig = plt.figure()
            ax = lamarckian.util.mpl.gca(fig, self.dim)
            improve = np.array([individual['result']['fitness'] - individual['_result']['fitness'] for individual in self.offspring])
            ax.set_title(f'improve={sum(improve > 0)}/{len(improve)}')
            cmap = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(np.min(improve), np.max(improve)), cmap=matplotlib.cm.jet)
            for individual, color in zip(self.offspring, cmap.to_rgba(improve)):
                if individual['result']['fitness'] > individual['_result']['fitness']:
                    linestyle = '-'
                else:
                    linestyle = ':'
                ax.arrow(*individual['_result']['objective'], *(np.array(individual['result']['objective']) - individual['_result']['objective']), color=color, linestyle=linestyle, alpha=0.3)
            color = np.array([individual['_result']['fitness'] for individual in self.offspring] + [individual['result']['fitness'] for individual in self.offspring])
            cmap = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(np.min(color), np.max(color)), cmap=matplotlib.cm.jet)
            ax.scatter(*zip(*[individual['result']['objective'] for individual in self.offspring]), c=[cmap.to_rgba(individual['result']['fitness']) for individual in self.offspring], marker='o')
            # ax.scatter(*zip(*[individual['_result']['objective'] for individual in self.offspring]), c=[cmap.to_rgba(individual['_result']['fitness']) for individual in self.offspring], marker='.')
            fig.canvas.draw()
            image = lamarckian.util.mpl.to_image(fig)
            prefix = os.path.join(self.kwargs['root'], self.tag, str(self.cost))
            os.makedirs(os.path.dirname(prefix), exist_ok=True)
            fig.savefig(prefix + '.svg')
            torch.save(self.offspring, prefix + '.pth')
            plt.close(fig)
            recorder.writer.add_image(self.tag, np.transpose(image, [2, 0, 1]), self.cost)
            # recorder.writer.add_scalar(f'{self.tag}/improve', sum(improve > 0) / len(improve), self.cost)
