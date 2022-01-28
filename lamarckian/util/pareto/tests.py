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

import inspect
import importlib


def test_dominate_min():
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function[len('test_'):]
    dominate = getattr(importlib.import_module('.'.join(__name__.split('.')[:-1])), NAME_FUNC)
    assert dominate([0, 0], [0, 1])
    assert dominate([0, 0], [1, 0])
    assert dominate([0, 0], [1, 1])
    assert not dominate([0, 0], [0, 0])
    assert not dominate([0, 0], [1, -1])
    assert not dominate([0, 0], [-1, 1])
    assert not dominate([0, 0], [-1, -1])


def test_dominate_max():
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function[len('test_'):]
    dominate = getattr(importlib.import_module('.'.join(__name__.split('.')[:-1])), NAME_FUNC)
    assert dominate([0, 0], [0, -1])
    assert dominate([0, 0], [-1, 0])
    assert dominate([0, 0], [-1, -1])
    assert not dominate([0, 0], [0, 0])
    assert not dominate([0, 0], [-1, 1])
    assert not dominate([0, 0], [1, -1])
    assert not dominate([0, 0], [1, 1])
