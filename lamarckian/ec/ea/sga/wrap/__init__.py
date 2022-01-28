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

import inspect
import logging

import glom


def elite(ea):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    def reevaluate(self, elite):
        variation = [self.rpc_async.receive()]
        self.rpc_async.send('evaluation', [{**elite, **{PATH_FUNC: None}}])
        while True:
            offspring = self.rpc_async.receive()
            if len(offspring) == 1 and PATH_FUNC in offspring[0]:
                elite, = offspring
                break
            else:
                variation.append(offspring)
        for offspring in variation:
            self.rpc_async.send(None, offspring)
        return elite

    class EA(ea):
        def evaluation0(self, *args, **kwargs):
            population = super().evaluation0(*args, **kwargs)
            setattr(self, NAME_FUNC, max(population, key=lambda individual: individual['result']['fitness']))
            return population

        def __setstate__(self, state):
            state = super().__setstate__(state)
            setattr(self, NAME_FUNC, max(self.population, key=lambda individual: individual['result']['fitness']))
            return state

        def selection(self, population):
            if glom.glom(self.kwargs['config'], 'ec.reevaluate', default=False):
                setattr(self, NAME_FUNC, reevaluate(self, getattr(self, NAME_FUNC)))
            return super().selection(population + [getattr(self, NAME_FUNC)])

        def __call__(self, *args, **kwargs):
            outcome = super().__call__(*args, **kwargs)
            elite = max(self.population, key=lambda individual: individual['result']['fitness'])
            if elite['result']['fitness'] > getattr(self, NAME_FUNC)['result']['fitness']:
                setattr(self, NAME_FUNC, elite)
            return outcome
    return EA
