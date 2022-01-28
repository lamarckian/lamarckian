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

import hashlib
import pickle
import warnings

from . import evaluator, record, mdp


def check(evaluator):
    class Evaluator(evaluator):
        def initialization(self, *args, **kwargs):
            population = super().initialization(*args, **kwargs)
            for key in population[0]['decision']:
                digests = [hashlib.md5(pickle.dumps(individual['decision'][key])).hexdigest() for individual in population]
                if len(set(digests)) < len(digests):
                    warnings.warn('\n'.join([key] + digests))
            return population

        def __next__(self):
            offspring = super().__next__()
            for individual in offspring:
                assert hashlib.md5(pickle.dumps(individual['decision'])).hexdigest() == individual['digest']
            return offspring

        def __call__(self, *args, **kwargs):
            for individual in self.population:
                assert hashlib.md5(pickle.dumps(individual['decision'])).hexdigest() == individual['digest']
            outcome = super().__call__(*args, **kwargs)
            for individual in self.population:
                assert hashlib.md5(pickle.dumps(individual['decision'])).hexdigest() == individual['digest']
            return outcome
    return Evaluator
