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

import inspect


def async_call(name):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    def wrap(cls):
        class Cls(cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, PATH_FUNC)

            def __call__(self, *args, **kwargs):
                data = super().__call__(*args, **kwargs)
                setattr(self, PATH_FUNC, data)
                return data

            def get_hooked(self):
                return getattr(self, PATH_FUNC)
        return Cls

    def decorate(rl):
        class RL(rl):
            def training(self, *args, **kwargs):
                training = super().training(*args, **kwargs)
                getattr(self, name).__class__ = wrap(type(getattr(self, name)))
                return training
        return RL
    return decorate
