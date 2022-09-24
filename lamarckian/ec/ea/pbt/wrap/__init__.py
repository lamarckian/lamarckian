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
import collections
import logging

import numpy as np
import scipy.stats
import glom

import lamarckian
from . import exploit


def opponent(ea):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    class EA(ea):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            setattr(self, PATH_FUNC, types.SimpleNamespace(
                enemies=glom.glom(kwargs['config'], 'rl.enemies', default=[1]),
                method=glom.glom(kwargs['config'], 'ec.ea.pbt.opponent.train', default='set'),
                update_eval=eval(f"lambda individual, count: " + glom.glom(kwargs['config'], 'ec.ea.pbt.opponent.eval', default="individual['result']['win'] > 0.9")),
                count=0,
            ))

        def evaluation0(self, population, *args, **kwargs):
            attr = getattr(self, PATH_FUNC)
            opponents = [{enemy: individual['decision']['blob'] for enemy in attr.enemies} for individual in population]
            self.rpc_all('set_opponents_train', opponents)
            self.rpc_all('set_opponents_eval', opponents)
            return super().evaluation0(population, *args, **kwargs)

        def get_variation_param(self, individual, *args, **kwargs):
            attr = getattr(self, PATH_FUNC)
            args, kwargs = super().get_variation_param(individual, *args, **kwargs)
            if 'call' not in kwargs:
                kwargs['call'] = {}
            opponents = [{enemy: individual['decision']['blob'] for enemy in attr.enemies} for individual in self.population]
            kwargs['call'][f"{attr.method}_opponents_train"] = (opponents,), {}
            if attr.update_eval(individual, attr.count):
                kwargs['call']['set_opponents_eval'] = (opponents,), {}
                logging.info('update opponents (eval)')
                attr.count = 0
            else:
                attr.count += 1
            return args, kwargs
    return EA


def safe(ea):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    class EA(ea):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            setattr(self, PATH_FUNC, types.SimpleNamespace(
                pvalue=glom.glom(kwargs['config'], 'pvalue', default=0.05),
                drop=collections.deque(maxlen=glom.glom(kwargs['config'], 'sample.train')),
            ))
            self.recorder.register(lamarckian.util.counter.Time(**glom.glom(self.kwargs['config'], 'record.plot')), lambda *args, **kwargs: lamarckian.util.record.Scalar(
                self.cost, **{
                    'pbt/drop': np.mean(getattr(self, PATH_FUNC).drop) if getattr(self, PATH_FUNC).drop else 0,
                },
            ))

        def apply(self, i, child):
            attr = getattr(self, PATH_FUNC)
            parent = self.population[i]
            drop = parent['result']['fitness'] > child['result']['fitness'] and scipy.stats.ttest_ind([result['fitness'] for result in parent['results']], [result['fitness'] for result in child['results']], equal_var=False)[1] < attr.pvalue
            attr.drop.append(drop)
            if not drop:
                self.population[i] = child
    return EA
