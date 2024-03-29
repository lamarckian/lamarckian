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

import itertools


def repeat_last(*numbers):
    def decorate(counter):
        class Counter(object):
            def __init__(self, *args, **kwargs):
                self.numbers = itertools.chain(numbers, itertools.repeat(numbers[-1]))
                self.make = lambda: counter(next(self.numbers), *args, **kwargs)
                self.counter = self.make()

            def __call__(self, *args, **kwargs):
                if self.counter(*args, **kwargs):
                    self.counter = self.make()
                    return True
                else:
                    return False

            def __getstate__(self):
                return dict(numbers=self.numbers, counter=self.counter)

            def __setstate__(self, state):
                self.numbers = state['numbers']
                self.counter = state['counter']
        return Counter
    return decorate
