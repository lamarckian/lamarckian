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

from timeit import default_timer as timer


stats = {}


def wrap(name, stats=stats):
    def decorate(func):
        def _(*args, **kw):
            start = timer()
            try:
                return func(*args, **kw)
            finally:
                stats[name] = timer() - start
        return _
    return decorate


class Measure(object):
    def __init__(self):
        self.start = timer()

    def close(self):
        self.duration = timer() - self.start

    def __call__(self):
        return self.duration
