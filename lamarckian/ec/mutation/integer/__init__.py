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
import collections.abc

import numpy as np

CODING = os.path.basename(os.path.dirname(__file__))


class Mutation(object):
    def __init__(self, encoding, **kwargs):
        self.kwargs = kwargs
        if isinstance(encoding, collections.abc.Mapping):
            self.lower, self.upper = np.stack(list(encoding.values())).T
        elif isinstance(encoding, np.ndarray):
            self.lower, self.upper = encoding.T
        else:
            raise TypeError(type(encoding))
        assert len(self.lower.shape) == 1, self.lower.shape
        assert len(self.upper.shape) == 1, self.upper.shape
        assert np.all(self.lower < self.upper), (self.lower, self.upper)

    def close(self):
        pass

from . import bitwise
