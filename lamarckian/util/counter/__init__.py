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

import time

import humanfriendly

from . import wrap


class Number(object):
    def __init__(self, max, first=0, **kwargs):
        self.max = max
        self.count = max if first is None else first
        self.kwargs = kwargs

    def __repr__(self):
        return f'{self.count}/{self.max}'

    def __int__(self):
        return self.count

    def __len__(self):
        return self.max

    def reset(self, count=0):
        self.count = count

    def __call__(self, step=1):
        self.count += step
        if self.count >= self.max:
            self.count = 0
            return True
        return False


class Time(object):
    def __init__(self, interval, first=False, **kwargs):
        self.start = 0 if first else time.time()
        self.interval = humanfriendly.parse_timespan(interval) if isinstance(interval, str) else interval
        self.kwargs = kwargs

    def __repr__(self):
        elapsed = time.time() - self.start
        return f'{max(self.interval - elapsed, 0)}'

    def __call__(self):
        t = time.time()
        elapsed = t - self.start
        if elapsed > self.interval:
            self.start = t
            return True
        else:
            return False

    def reset(self):
        self.start = time.time()
