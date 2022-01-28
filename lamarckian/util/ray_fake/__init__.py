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
import types

from . import services, util


def put(x):
    return x


def get(x):
    return x


def wait(ids, num_returns=1, **kwargs):
    return ids[:num_returns], ids[num_returns:]


def actors(id):
    return {}


def nodes():
    return [dict(Resources=dict(CPU=1))]


class Actor(object):
    def __init__(self, instance):
        self._ = instance
        for name, func in inspect.getmembers(instance, predicate=inspect.isroutine):
            setattr(self, name, types.SimpleNamespace(remote=func))
        self._actor_id = types.SimpleNamespace(hex=lambda: '')

    def __hash__(self):
        return id(self)


def remote(*args, **kwargs):
    def decorate(cls):
        def remote(*args, **kwargs):
            return Actor(cls(*args, **kwargs))
        return types.SimpleNamespace(remote=remote, options=lambda *args, **kwargs: types.SimpleNamespace(remote=remote))
    return decorate
