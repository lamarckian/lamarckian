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
import torch
import glom

import lamarckian


def get_feature(self):
    ball = self.mdp.get_ball()
    bat = self.get_position_bat()
    return torch.Tensor(np.array([ball._speed_x > 0, bat]))


def immediate(mdp):
    @lamarckian.mdp.wrap.reward.blob.immediate(get_feature)
    class MDP(mdp):
        def describe_reward(self):
            return dict(
                model=dict(
                    cls=glom.glom(self.kwargs['config'], f'mdp.pong.reward.model') + self.kwargs['config']['model']['wrap'],
                    kwargs=dict(inputs=4),
                ),
            )
    return MDP


def potential(mdp):
    @lamarckian.mdp.wrap.reward.blob.potential(get_feature)
    class MDP(mdp):
        def describe_reward(self):
            return dict(
                model=dict(
                    cls=glom.glom(self.kwargs, f'mdp.pong.reward.model') + self.kwargs['config']['model']['wrap'],
                    kwargs=dict(inputs=2),
                ),
            )
    return MDP
