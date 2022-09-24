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
import contextlib

import glom

import lamarckian
from ... import util


def all(actor):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    def wrap(executor):
        def decorate(forward):
            def _(*args, **kwargs):
                executor.lock()
                with contextlib.closing(types.SimpleNamespace(close=executor.unlock)):
                    return forward(*args, **kwargs)
            return _
        return decorate

    def setup_execute(self, param, result, **kwargs):
        import pylamarckian
        executor = pylamarckian.rl.ppo.broadcaster.Executor(
            [p.data for p in self.model.parameters()],
            param, result,
            glom.glom(kwargs['config'], 'rpc.serializer'),
            kwargs['shm'], kwargs['shm_name'],
            glom.glom(kwargs['config'], 'rl.ppo.fp16', default=False),
        )
        self.model.forward = wrap(executor)(self.model.forward)
        setattr(self, PATH_FUNC, executor)

    class Actor(lamarckian.util.rpc.wrap.all(actor)):
        def close(self):
            super().close()
            if hasattr(self, PATH_FUNC):
                getattr(self, PATH_FUNC).join()

        def setup_rpc_all(self, mode=None, *args, **kwargs):
            if 'ray' not in kwargs and mode == 'execute' and util.broadcaster.__name__ in kwargs and glom.glom(kwargs['config'], 'c.executor', default=True):
                try:
                    return setup_execute(self, *args, **kwargs)
                except ImportError:
                    pass
            return super().setup_rpc_all(mode, *args, **kwargs)

        def get_iteration(self):
            try:
                return getattr(self, PATH_FUNC).iteration
            except AttributeError:
                return super().get_iteration()
    return Actor
