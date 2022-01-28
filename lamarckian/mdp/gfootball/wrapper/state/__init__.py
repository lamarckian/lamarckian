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

import numbers

import gym
import numpy as np
from gfootball.env.wrappers import Simple115StateWrapper

from lamarckian.mdp.gfootball import util
from .kaggle_helper import Action, GameMode


def one_hot_owner(observation):
    team, player = observation['ball_owned_team'], observation['ball_owned_player']
    if team == -1:  # none
        return [0 for _ in range(22)] + [1]
    elif team == 0:  # left
        return util.one_hot(23, player)
    else:  # right
        if player == -1:
            return util.one_hot(23, -1)
        else:
            return util.one_hot(23, player + 12)


def feature_angle(targets, base):
    cos = targets.dot(base) / np.maximum(np.linalg.norm(targets, axis=-1) * np.linalg.norm(base), np.finfo(targets.dtype).eps)
    sin = np.sqrt(np.maximum(1 - cos ** 2, 0))
    mask = base[1] - targets[:, 1] < 0
    sin[mask] = -sin[mask]
    return np.stack([cos, sin], axis=-1)


class State(gym.ObservationWrapper):
    boundary = np.array([(-1, -0.42), (1, 0.42)])

    @classmethod
    def get_header(cls):
        raise NotImplementedError()

    def __init__(self, env, *args, **kwargs):
        super().__init__(env)
        self.env.unwrapped.header_state = self.get_header()

    @classmethod
    def convert_observation1(cls, observation):
        raise NotImplementedError()

    @classmethod
    def convert_observation(cls, observations):
        return [cls.convert_observation1(observation) for observation in observations]

    def observation(self, observation):
        states = self.convert_observation(observation)
        for inputs in states:
            for header, input in zip(self.env.unwrapped.header_state, inputs):
                shape = tuple(x if isinstance(x, numbers.Number) else len(x) for x in header)
                assert shape == input.shape, (shape, input.shape)
        return states


class Simple115(State):
    @classmethod
    def get_header_player(cls):
        """
        https://github.com/google-research/football/blob/master/gfootball/doc/observation.md
        22 - (x,y) coordinates of left team players
        22 - (x,y) direction of left team players
        22 - (x,y) coordinates of right team players
        22 - (x, y) direction of right team players
        3 - (x, y and z) - ball position
        3 - ball direction
        3 - one hot encoding of ball ownership (noone, left, right)
        11 - one hot encoding of which player is active
        7 - one hot encoding of game_mode
        """
        header = []
        for i in range(11):
            header += [f'x{i}', f'y{i}']
        for i in range(11):
            header += [f'dx{i}', f'dy{i}']
        for i in range(11):
            header += [f'_x{i}', f'_y{i}']
        for i in range(11):
            header += [f'_dx{i}', f'_dy{i}']
        header += ['x', 'y', 'z']
        header += ['dx', 'dy', 'dz']
        header += [f'owner_{key}' for key in util.Owner.__members__.keys()]
        for i in range(11):
            header.append(f'active{i}')
        header += list(util.Mode.__members__.keys())
        assert len(header) == 115, len(header)
        return [header]

    @classmethod
    def get_header(cls):
        return [
            cls.get_header_player(),
        ]

    @classmethod
    def convert_observation1_player(cls, observation):
        state, = Simple115StateWrapper.convert_observation([observation], False)
        return state

    @classmethod
    def convert_observation1(cls, observation):
        return [
            cls.convert_observation1_player(observation),
        ]
