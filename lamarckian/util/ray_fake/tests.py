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

import lamarckian.util.ray_fake as ray


class Actor(object):
    def __init__(self, a):
        self.a = a

    def test0(self):
        pass

    def test1(self, b):
        self.b = b
        return b

    def __call__(self, *args, **kwargs):
        return args, kwargs


def test():
    actor = ray.remote()(Actor).remote(0)
    assert ray.get(actor.test0.remote()) is None
    assert ray.get([actor.test1.remote(1), actor.test1.remote(2)]) == [1, 2]
    assert ray.get(actor.__call__.remote(3, 4)) == ((3, 4), dict())
