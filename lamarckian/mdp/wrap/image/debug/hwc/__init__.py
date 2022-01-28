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
import time
import inspect

import numpy as np
import imageio
import matplotlib.pyplot as plt
import seaborn as sns


def bitmap(index, name=None, root=os.path.join('~', 'debug', 'gym'), ext='.bmp'):
    if name is None:
        name = inspect.getframeinfo(inspect.currentframe()).function
    root = os.path.join(os.path.expanduser(os.path.expandvars(root)), time.strftime('%Y-%m-%d_%H-%M-%S.%f'), name)

    def decorate(mdp):
        class MDP(mdp):
            class Controller(mdp.Controller):
                def get_state(self):
                    state = super().get_state()
                    image = state['inputs'][index]
                    os.makedirs(root, exist_ok=True)
                    imageio.imwrite(os.path.join(root, f'{self.mdp.frame}{ext}'), image)
                    return state
        return MDP
    return decorate


def bitmap_stack(index, name=None, root=os.path.join('~', 'debug', 'gym'), ext='.bmp'):
    if name is None:
        name = inspect.getframeinfo(inspect.currentframe()).function
    root = os.path.join(os.path.expanduser(os.path.expandvars(root)), time.strftime('%Y-%m-%d_%H-%M-%S.%f'), name)

    def decorate(mdp):
        class MDP(mdp):
            class Controller(mdp.Controller):
                def get_state(self):
                    state = super().get_state()
                    image = state['inputs'][index]
                    assert len(image.shape) == 3, image.shape
                    os.makedirs(root, exist_ok=True)
                    images = np.array_split(image, image.shape[-1], axis=-1)
                    imageio.imwrite(os.path.join(root, f'{self.mdp.frame}{ext}'), np.concatenate(images, axis=1))
                    imageio.imwrite(os.path.join(root, f'_{self.mdp.frame}{ext}'), np.concatenate([image1 - image2 for image1, image2 in zip(images[:-1], images[1:])], axis=1))
                    return state
        return MDP
    return decorate


def heatmap(index, name=None, root=os.path.join('~', 'debug', 'gym'), ext='.svg', **kwargs):
    if name is None:
        name = inspect.getframeinfo(inspect.currentframe()).function
    root = os.path.join(os.path.expanduser(os.path.expandvars(root)), time.strftime('%Y-%m-%d_%H-%M-%S.%f'), name)

    def decorate(mdp):
        class MDP(mdp):
            class Controller(mdp.Controller):
                def get_state(self):
                    state = super().get_state()
                    image = state['inputs'][index]
                    fig = plt.figure()
                    ax = fig.gca()
                    sns.heatmap(image, ax=ax, **kwargs)
                    fig.tight_layout()
                    os.makedirs(root, exist_ok=True)
                    fig.savefig(os.path.join(root, f'{self.mdp.frame}{ext}'))
                    plt.close(fig)
                    return state
        return MDP
    return decorate
