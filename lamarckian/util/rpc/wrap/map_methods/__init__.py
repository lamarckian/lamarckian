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


def all(*keys, rpc=lambda self: self.rpc_all, local=False):
    def call(key):
        def _(self, *args, **kwargs):
            return rpc(self)(key, *args, **kwargs)
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


def any(*keys, rpc=lambda self: self.rpc_any):
    def call(self, key, *args, **kwargs):
        return rpc(self)(key, *args, **kwargs)

    def decorate(cls):
        for key in keys:
            setattr(cls, key, functools.partialmethod(call, key))
        return cls
    return decorate
