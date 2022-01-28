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

import os
import inspect
import hashlib
import types
import time

import numpy as np
import torch
import humanfriendly
import glom

import lamarckian


def csv(relpath='stopper/csv', keep=0, model=False, interval='stopper.debug.interval'):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = hashlib.md5((__file__ + NAME_FUNC + relpath).encode()).hexdigest()

    def decorate(stopper):
        class Stopper(stopper):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, PATH_FUNC)
                if kwargs.get('root', None) is not None:
                    root = os.path.expanduser(os.path.expandvars(os.path.join(kwargs['root'], relpath)))
                    os.makedirs(root, exist_ok=True)
                    attr = types.SimpleNamespace(
                        table=[('Wall time', 'Step', 'Value')],
                        prefix=os.path.join(root, str(self.evaluator.cost)),
                        interval=lamarckian.util.counter.Time(humanfriendly.parse_timespan(glom.glom(kwargs['config'], interval))) if isinstance(interval, str) else lambda: False,
                    )
                    if model:
                        torch.save(self.evaluator.__getstate__(), attr.prefix + '.pth')
                    setattr(self, PATH_FUNC, attr)

            def close(self):
                if hasattr(self, PATH_FUNC):
                    attr = getattr(self, PATH_FUNC)
                    np.savetxt(attr.prefix + '.csv', attr.table, fmt='%s', delimiter=',')
                    if keep > 0:
                        lamarckian.util.file.tidy(os.path.dirname(attr.prefix), keep)
                return super().close()

            def __call__(self, outcome, *args, **kwargs):
                if hasattr(self, PATH_FUNC):
                    attr = getattr(self, PATH_FUNC)
                    for result in outcome['results']:
                        attr.table.append((time.time(), self.evaluator.cost, result['fitness']))
                    if attr.interval():
                        np.savetxt(attr.prefix + '.csv', attr.table, fmt='%s', delimiter=',')
                return super().__call__(outcome, *args, **kwargs)
        return Stopper
    return decorate
