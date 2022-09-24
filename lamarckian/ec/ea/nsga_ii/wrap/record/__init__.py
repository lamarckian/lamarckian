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

import glom

import lamarckian


def nds(tag):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    def decorate(ea):
        class EA(ea):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, PATH_FUNC)
                self.recorder.register(lamarckian.util.counter.Time(**glom.glom(kwargs['config'], 'record.scalar')), lambda *args, **kwargs: lamarckian.util.record.Scalar(self.cost, **{f'{tag}/{key}': value for key, value in getattr(self, PATH_FUNC).items()}))

            def __call__(self, *args, **kwargs):
                setattr(self, PATH_FUNC, dict(layers=0))
                return super().__call__(*args, **kwargs)

            def assign_non_critical(self, *args, **kwargs):
                getattr(self, PATH_FUNC)['layers'] += 1
                return super().assign_non_critical(*args, **kwargs)

            def select_critical(self, *args, **kwargs):
                population = super().select_critical(*args, **kwargs)
                attr = getattr(self, PATH_FUNC)
                attr['layers'] += 1
                attr['critical'] = len(population)
                return population
        return EA
    return decorate
