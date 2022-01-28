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

import numpy as np

from . import Shubert


def test_shubert():
    evaluator = Shubert()
    evaluator.set(dict(real=np.array([-1.4250, -0.8003])))
    result, = evaluator.evaluate()
    np.testing.assert_almost_equal(result['fitness'], 186.73086940373756)
