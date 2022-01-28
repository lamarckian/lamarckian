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

import functools

import ray


def any(*keys, force=False, get_actor=lambda self: self.receive.get_idle() if hasattr(self, 'receive') else self.actors[0]):
    def call(self, key, *args, **kwargs):
        _ray = self.kwargs.get('ray', ray)
        actor = get_actor(self)
        return _ray.get(getattr(actor, key).remote(*args, **kwargs))

    def decorate(cls):
        for key in keys:
            if force or not hasattr(cls, key):
                setattr(cls, key, functools.partialmethod(call, key))
        return cls
    return decorate


def all(*keys, local=True, get_actors=lambda self: self.actors):
    def call(key):
        def _(self, *args, **kwargs):
            _ray = self.kwargs.get('ray', ray)
            return _ray.get([getattr(actor, key).remote(*args, **kwargs) for actor in get_actors(self)])[0]
        return _

    def merge(*locals):
        def decorate(remote):
            def _(*args, **kwargs):
                for func in locals:
                    func(*args, **kwargs)
                return remote(*args, **kwargs)
            return _
        return decorate

    def decorate(cls):
        for key in keys:
            func = call(key)
            if local and hasattr(cls, key):
                func = merge(getattr(cls, key))(func)
            setattr(cls, key, func)
        return cls
    return decorate
