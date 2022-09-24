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

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

import lamarckian


class Nash(object):
    def __init__(self, tag, cost, payoff, layer=None, root=None):
        self.tag = tag
        self.cost = cost
        self.payoff = payoff
        self.layer = layer
        self.root = root

    def dump(self):
        if self.root is not None:
            path = os.path.join(self.root, self.tag, f"{self.cost}.pth")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self.payoff, path)

    def __call__(self, recorder):
        self.dump()
        fig = plt.figure()
        ax = fig.gca()
        if self.layer is None:
            label = False
        else:
            label = ['' for _ in range(len(self.payoff))]
            for i, prob in self.layer.items():
                label[i] = format(prob, '.2f')
        limit = np.abs(self.payoff).max()
        sns.heatmap(self.payoff, vmin=-limit, vmax=limit, cmap=plt.get_cmap('seismic'), cbar=False, xticklabels=False, yticklabels=label, square=True, ax=ax)
        fig.tight_layout()
        fig.canvas.draw()
        image = lamarckian.util.mpl.to_image(fig)
        recorder.writer.add_image(self.tag, np.transpose(image, [2, 0, 1]), self.cost)
        plt.close(fig)