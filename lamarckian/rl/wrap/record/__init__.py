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
import psutil

import lamarckian


def memory(rl):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function

    import tracemalloc
    tracemalloc.start()

    def get(process):
        current, peak = tracemalloc.get_traced_memory()
        return {
            f"{NAME_FUNC}/rss": process.memory_info().rss,
            f"{NAME_FUNC}/tracemalloc/current": current,
            f"{NAME_FUNC}/tracemalloc/peak": peak,
        }

    class RL(rl):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            process = psutil.Process()
            self.recorder.register(
                lamarckian.util.counter.Time(**glom.glom(kwargs['config'], 'record.scalar')),
                lambda outcome: lamarckian.util.record.Scalar(self.cost, **get(process)),
            )
            top = glom.glom(kwargs['config'], f"{NAME_FUNC}.tracemalloc", default=10)
            self.recorder.register(
                lamarckian.util.counter.Time(**glom.glom(kwargs['config'], 'record.scalar')),
                lambda *args, **kwargs: lamarckian.util.record.Text(self.cost, **{f"{NAME_FUNC}/tracemalloc/top": '\n\n'.join([str(stat) for stat in tracemalloc.take_snapshot().statistics('lineno')[:top]])}),
            )
    return RL
