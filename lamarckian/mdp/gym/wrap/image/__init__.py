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

import inspect
import collections

import numpy as np

from . import color, debug


def crop(ymin, ymax, xmin, xmax):
    def make(image):
        if len(image.shape) == 2:
            return image[ymin:ymax, xmin:xmax]
        else:
            return image[ymin:ymax, xmin:xmax, :]

    def decorate(mdp):
        class MDP(mdp):
            class Controller(mdp.Controller):
                def get_state(self):
                    state = super().get_state()
                    state['inputs'] = tuple(make(input) for input in state['inputs'])
                    return state

            def describe_blob(self):
                encoding = super().describe_blob()
                for model in encoding['models']:
                    for input in model['inputs']:
                        input['shape'] = make(np.zeros(input['shape'])).shape
                return encoding
        return MDP
    return decorate


def downsample(height, width):
    def make(image):
        if len(image.shape) == 2:
            return image[::height, ::width]
        else:
            return image[::height, ::width, :]

    def decorate(mdp):
        class MDP(mdp):
            class Controller(mdp.Controller):
                def get_state(self):
                    state = super().get_state()
                    state['inputs'] = tuple(make(input) for input in state['inputs'])
                    return state

            def describe_blob(self):
                encoding = super().describe_blob()
                for model in encoding['models']:
                    for input in model['inputs']:
                        input['shape'] = make(np.zeros(input['shape'])).shape
                return encoding
        return MDP
    return decorate


def resize(height, width, **kwargs):
    import cv2

    def make(image):
        return cv2.resize(image, (width, height), **kwargs)

    def decorate(mdp):
        class MDP(mdp):
            class Controller(mdp.Controller):
                def get_state(self):
                    state = super().get_state()
                    state['inputs'] = tuple(make(input) for input in state['inputs'])
                    return state

            def describe_blob(self):
                encoding = super().describe_blob()
                for model in encoding['models']:
                    for input in model['inputs']:
                        input['shape'] = make(np.zeros(input['shape'])).shape
                return encoding
        return MDP
    return decorate


def stack(size):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function

    def make(self, image):
        if not hasattr(self, NAME_FUNC):
            setattr(self, NAME_FUNC, collections.deque([image] * size, maxlen=size))
        attr = getattr(self, NAME_FUNC)
        attr.append(image)
        assert len(attr) == size
        return np.concatenate(attr, axis=0)

    def decorate(mdp):
        class MDP(mdp):
            class Controller(mdp.Controller):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    assert not hasattr(self, NAME_FUNC)

                def get_state(self):
                    state = super().get_state()
                    state['inputs'] = tuple(make(self, input) for input in state['inputs'])
                    return state

            def describe_blob(self):
                encoding = super().describe_blob()
                for model in encoding['models']:
                    for input in model['inputs']:
                        input['shape'] = (input['shape'][0] * size,) + input['shape'][1:]
                return encoding
        return MDP
    return decorate
