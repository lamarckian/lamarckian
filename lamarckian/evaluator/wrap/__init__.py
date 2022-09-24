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
import types
import traceback

from . import remote


def fix_ray(evaluator):
    class Evaluator(evaluator):
        def __init__(self, *args, **kwargs):
            try:
                super().__init__(*args, **kwargs)
            except:
                traceback.print_exc()
                raise
    return Evaluator


def nonstop(evaluator):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    class Evaluator(evaluator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)

        def close(self):
            if hasattr(self, PATH_FUNC):
                getattr(self, PATH_FUNC).close()
            return super().close()

        def training(self):
            if not hasattr(self, PATH_FUNC):
                setattr(self, PATH_FUNC, super().training())
            return types.SimpleNamespace(close=lambda: None)
    return Evaluator
