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

import numpy as np
import mpl_toolkits.mplot3d


def gca(fig, dim):
    if dim == 3:
        ax = mpl_toolkits.mplot3d.Axes3D(fig)
        for name in ['x', 'y', 'z']:
            getattr(ax, f'set_{name}label')(name)
    else:
        ax = fig.gca()
    return ax


def to_image(fig):
    w, h = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape([h, w, -1])
    return image[:, :, 1:]
