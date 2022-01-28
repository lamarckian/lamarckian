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

import torch

from . import Action


def rule(mdp, me):
    ball = mdp.get_ball()
    bat = mdp.get_bats()[me]
    if bat._rect.centery < ball.centery:
        return Action.down
    else:
        return Action.up


class Rule(object):
    def __init__(self, mdp, me):
        self.mdp = mdp
        self.me = me

    def __call__(self, state):
        action = rule(self.mdp, self.me)
        return dict(action=torch.LongTensor([action.value]))
