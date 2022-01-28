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

import lamarckian
import glom

from . import real, integer, blob, wrap


class Useless(object):
    def __init__(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def __call__(self, individual, **kwargs):
        return individual


class Mutation(object):
    def __init__(self, encoding, **kwargs):
        config = glom.glom(kwargs['config'], 'ec.mutation')
        self.mutation = {}
        for key, encoding in encoding.items():
            _kwargs = {**kwargs, **config[key]}
            cls = lamarckian.evaluator.parse(*_kwargs['create'])
            self.mutation[key] = cls(encoding, **_kwargs)

    def close(self):
        for mutation in self.mutation.values():
            mutation.close()

    def __call__(self, individual):
        for key, value in individual['decision'].items():
            individual['decision'][key] = self.mutation[key](value, **individual.get('result', {}))
        return individual
