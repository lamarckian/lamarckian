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

import itertools
import operator
import re


def rsplit(n):
    def _(model):
        return [tuple(map(operator.itemgetter(0), items)) for prefix, items in itertools.groupby(enumerate(model.state_dict().keys()), lambda item: item[1].rsplit('.', n)[0])]
    return _


def last_digits(model):
    prog = re.compile(r'(^.*\.\d+)')

    def prefix(item):
        m = prog.search(item[1])
        if m is None:
            return m
        else:
            return m.group(0)
    return [tuple(map(operator.itemgetter(0), items)) for prefix, items in itertools.groupby(enumerate(model.state_dict().keys()), prefix)]
