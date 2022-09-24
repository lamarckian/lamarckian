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

import os

import numpy as np

from . import evaluator

CODING = os.path.basename(os.path.dirname(__file__))


def blob(coding=CODING):
    def decorate(ea):
        class EA(ea):
            def get_center(self, *args, **kwargs):
                center = super().get_center(*args, **kwargs)
                assert coding not in center
                center[coding] = [np.mean([particle['decision'][coding][index] for particle in self.population], 0) for index in range(len(self.population[0]['decision'][coding]))]
                return center
        return EA
    return decorate


def real(coding=CODING):
    def decorate(ea):
        class EA(ea):
            def get_center(self, *args, **kwargs):
                center = super().get_center(*args, **kwargs)
                assert coding not in center
                center[coding] = np.mean([particle['decision'][coding] for particle in self.population], 0)
                return center
        return EA
    return decorate
