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

import numpy as np
import matplotlib.pyplot as plt

import lamarckian
from lamarckian.mdp.gfootball import util


class Rollout(lamarckian.rl.record.Rollout):
    def dump(self, recorder, prefix, trajectory, exp, result, **kwargs):
        super().dump(recorder, prefix, trajectory, exp, result, **kwargs)
        if 'observation' in exp['state']:
            fig = plt.figure()
            ax = fig.gca()
            observations = [exp['state']['observation'] for exp in trajectory + [exp]]
            points = [observation['ball'][:2] for observation in observations if observation['ball_owned_team'] == 0]
            if points:
                ax.plot(*zip(*points), 'o', color='red', alpha=0.1)
            points = [observation['ball'][:2] for observation in observations if observation['ball_owned_team'] == 1]
            if points:
                ax.plot(*zip(*points), 'o', color='blue', alpha=0.1)
            points = [observation['ball'][:2] for observation in observations if observation['ball_owned_team'] == -1]
            if points:
                ax.plot(*zip(*points), 'o', color='lightgrey', alpha=0.1)
            ax.set_xlim([-1, 1])
            ax.set_ylim([-util.Y_HALF, util.Y_HALF])
            ax.invert_yaxis()
            ax.set_aspect('equal')
            fig.tight_layout()
            plt.savefig(prefix + '.svg')
            fig.canvas.draw()
            image = lamarckian.util.mpl.to_image(fig)
            recorder.writer.add_image(f"gfootball/{os.path.basename(prefix)}", np.transpose(image, [2, 0, 1]), self.cost)
