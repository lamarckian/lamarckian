"""Copyright (C) 2020

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>."""

from scipy.special import comb
from itertools import combinations
import numpy as np
import inspect
import glom


class UniformReference(object):
    '''UniformPoint - Generate a set of uniformly distributed points on the unit hyperplane.

       [weight, point_num] = UniformPoint(point_num, obj_num) returns approximately point_num uniformly distributed points
       with obj_num objectives on the unit hyperplane.

        Due to the requirement of uniform distribution, the number of points cannot be arbitrary, and the number
        of points in weight may be slightly smaller than the predefined size point_num.

        Example:
            [weight, point_num] = UniformPoint(275,10)

----------------------------------------------- Reference --------------------------------------------------------------
    [1] I. Das and J. E. Dennis, Normal-boundary intersection: A new method for generating the Pareto surface
    in nonlinear multicriteria optimization problems, SIAM Journal on Optimization, 1998, 8(obj_num): 631-657.
    [2] K. Deb and H. Jain, An evolutionary many-objective optimization algorithm using reference-point based
    non-dominated sorting approach, part I: Solving problems with box constraints, IEEE Transactions on
    Evolutionary Computation, 2014, 18(4): 577-601.
    '''

    def uniform_reference(self, point_num, obj_num):
        h1 = 1
        while comb(h1 + obj_num, obj_num - 1) <= point_num:
            h1 = h1 + 1

        weight = np.array(list(combinations(range(1, h1 + obj_num), obj_num - 1))) - \
                 np.tile(np.array(range(obj_num - 1)), [int(comb(h1 + obj_num - 1, obj_num - 1)), 1]) - 1
        weight = (np.hstack((weight, np.zeros((weight.shape[0], 1)) + h1)) - np.hstack(
            (np.zeros((weight.shape[0], 1)), weight))) / h1
        if h1 < obj_num:
            h2 = 0
            while comb(h1 + obj_num - 1, obj_num - 1) + comb(h2 + obj_num, obj_num - 1) <= point_num:
                h2 = h2 + 1
            if h2 > 0:
                weight2 = np.array(list(combinations(range(1, h2 + obj_num - 1), obj_num - 1))) - \
                          np.tile(np.array(range(obj_num - 1)), [int(comb(h2 + obj_num - 1, obj_num - 1)), 1])
                weight2 = (np.hstack((weight2, np.zeros((weight2.shape[0], 1)) + h2)) - np.hstack(
                    (np.zeros((weight2.shape[0], 1)), weight2))) / h2
                weight = np.vstack((weight, weight2 / 2 + 1 / (2 * obj_num)))
        weight = np.maximum(weight, np.full((weight.shape[0], weight.shape[1]), 1e-6))
        point_num = weight.shape[0]
        return -weight, point_num

    def __init__(self, **kwargs):
        point_num = glom.glom(kwargs['config'], 'ec.population')
        # obj_num = glom.glom(kwargs['config']['benchmark'], 'dtlz.objective')              # how to dynamically obtain objective size?
        obj_num = 2;
        weights_total, weights_total_num = self.uniform_reference(point_num, obj_num)
        k = glom.glom(kwargs['config']['ec']['ea'], 'pocea.k')
        weights2 = weights_total[:weights_total_num:k, :]
        weights2_num = weights2.shape[0]
        self.weights = dict(weights=weights_total, weights_num=weights_total_num, sub_weights=weights2, sub_weights_num=weights2_num)

