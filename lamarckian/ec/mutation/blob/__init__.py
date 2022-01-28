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

import numpy as np
import glom

CODING = os.path.basename(os.path.dirname(__file__))


class Mutation(object):
    def __init__(self, encoding, **kwargs):
        self.encoding = encoding
        self.kwargs = kwargs

    def close(self):
        pass


class Gaussian(Mutation):
    def __init__(self, encoding, **kwargs):
        super().__init__(encoding, **kwargs)
        self.rs = np.random.RandomState(glom.glom(kwargs['config'], 'seed', default=None))
        self.stddev = kwargs['gaussian']

    def __call__(self, parent, **kwargs):
        return [value + self.rs.randn(*value.shape) * self.stddev for value in parent]
