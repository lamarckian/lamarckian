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

import pickle
import hashlib

import ray

from . import map_methods


def check_initialize(evaluator):
    class Evaluator(evaluator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            decisions = kwargs.get('ray', ray).get([actor.initialize.remote() for actor in self.actor])
            for key in decisions[0]:
                digests = [hashlib.md5(pickle.dumps(decision[key])).hexdigest() for decision in decisions]
                assert len(set(digests)) == len(digests), '\n'.join([key] + digests)
    return Evaluator
