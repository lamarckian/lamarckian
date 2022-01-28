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

import numpy as np
import matplotlib.pyplot as plt

import lamarckian
from lamarckian.ec.crossover.real import CODING
from lamarckian.ec.crossover.real.sbx import Crossover


def main():
    config = lamarckian.util.config.read(os.path.dirname(lamarckian.__file__) + '.yml')
    encoding = np.array([(-3, 5), (-1, 4)], np.float)
    crossover = Crossover(encoding, config=config, **{**config['crossover'][CODING], **dict(prob=1)})
    parent1 = np.array([-2, 3], np.float)
    parent2 = np.array([3, 0], np.float)
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(*zip(parent1, parent2), 'o', c='k')
    ax.text(*parent1, 'p1')
    ax.text(*parent2, 'p2')
    for _, prop in zip(range(500), itertools.cycle(plt.rcParams['axes.prop_cycle'])):
        child1, child2 = crossover(parent1, parent2)
        ax.plot(*zip(child1, child2), '.', c=prop['color'])
    fig.tight_layout()
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
