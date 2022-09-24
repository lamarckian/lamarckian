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

import glom

import lamarckian
from . import otherwise


def group(crossover):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function

    class Crossover(crossover):
        def __init__(self, encoding, *args, **kwargs):
            super().__init__(encoding, *args, **kwargs)
            assert not hasattr(self, NAME_FUNC)
            try:
                model = encoding['models'][encoding['me']]
            except KeyError:
                model = encoding['model']
            cls = lamarckian.evaluator.parse(*model['cls'], **kwargs)
            model = cls(**model, **kwargs)
            cls = lamarckian.evaluator.parse(*glom.glom(kwargs['config'], 'model.group'), **kwargs)
            setattr(self, NAME_FUNC, cls(model))
    return Crossover
