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


class Stopper(object):
    def __init__(self, evaluator, **kwargs):
        self.evaluator = evaluator
        self.kwargs = kwargs

    def close(self):
        pass

    def get(self):
        return self.evaluator.get()

    def evaluate(self):
        return self.evaluator.evaluate()


from . import cost, iteration, episode, fitness, objective, wrap


class Never(Stopper):
    def __call__(self, *args, **kwargs):
        return False


class Always(Stopper):
    def __call__(self, *args, **kwargs):
        return True
