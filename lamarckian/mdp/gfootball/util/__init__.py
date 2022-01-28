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
import math
import types
import enum
import itertools
import datetime
import binascii
import logging

import numpy as np
import matplotlib.pyplot as plt

PLAYERS = 11
Y_HALF = 0.42
GATE_HALF = 0.044
GOAL_AREA_X = 0.7
GOAL_AREA_HALF = 0.24
PLAYER_SIZE = 0.01
SHOT_RANGE = 0.3
GOAL = np.array([1, 0])


Owner = enum.Enum('Owner', [
    ('none', -1),
    ('me', 0),
    ('enemy', 1),
])
Mode = enum.Enum('Mode', zip('normal kickoff goal_kick free_kick corner throw_in penalty'.split(), itertools.count()))
Action = enum.Enum('Action', zip('idle left top_left top top_right right bottom_right bottom bottom_left long_pass high_pass short_pass shot sprint release_direction release_sprint sliding dribble release_dribble'.split(), itertools.count()))


def unit_vector(v):
    length = np.linalg.norm(v)
    if length:
        return v / length
    else:
        return np.array([1, 0])


def orthogonalize(v):
    angle = np.arctan2(*v[::-1]) + np.pi / 2
    return np.array([
        v,
        [np.cos(angle), np.sin(angle)],
    ])


def one_hot(size, index):
    value = [0] * size
    value[index] = 1
    return value


def get_axes(v):
    if np.linalg.norm(v) == 0:
        v = [1, 0]
    angle = np.arctan2(*v[::-1]) + np.pi / 2
    return np.array([
        v / np.linalg.norm(v), [math.cos(angle), math.sin(angle)]
    ])


def get_obstacle(ball, enemies, root=None):
    lower, upper = sorted([np.arctan2(y, x) for x, y in (np.array([1, -GATE_HALF]) - ball, np.array([1, GATE_HALF]) - ball)])
    _enemies = enemies - ball
    angles = np.arctan2(*_enemies.T[::-1])
    inside = np.logical_and(lower <= angles, angles <= upper)
    value = np.sum(inside)
    if root is not None:
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(*zip(ball, (1, -GATE_HALF)))
        ax.plot(*zip(ball, (1, GATE_HALF)))
        ax.scatter(*ball, marker='*', color='g')
        ax.scatter(*enemies[inside].T, color='r')
        ax.scatter(*enemies[~inside].T, color='b')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-Y_HALF, Y_HALF])
        ax.invert_yaxis()
        ax.set_title(str(value))
        ax.set_aspect('equal')
        fig.tight_layout()
        os.makedirs(root, exist_ok=True)
        path = os.path.join(root, f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')}_{binascii.b2a_hex(os.urandom(3)).decode()}.svg")
        plt.savefig(path)
        logging.info(path)
        plt.close(fig)
    return value


def get_obstacle_angle(ball, enemies, path=None):
    lower, upper = sorted([np.arctan2(y, x) for x, y in (np.array([1, -GATE_HALF]) - ball, np.array([1, GATE_HALF]) - ball)])
    _enemies = enemies - ball
    angles = np.arctan2(*_enemies.T[::-1])
    dist = np.linalg.norm(_enemies, axis=-1)
    inside = np.logical_and(lower <= angles, angles <= upper)
    # inside = np.logical_and(inside, dist < SHOT_RANGE)
    if np.any(inside):
        _dist = dist[inside]
        if np.any(_dist < PLAYER_SIZE):
            _lower, _upper = lower, upper
        else:
            _delta = np.arcsin(PLAYER_SIZE / _dist)
            _angles = angles[inside]
            _angles = sorted(np.concatenate([_angles - _delta, _angles + _delta]))
            _lower, _upper = np.clip(_angles[0], lower, upper), np.clip(_angles[-1], lower, upper)
    else:
        _lower, _upper = lower, lower
    value = (_lower - lower + upper - _upper) / np.radians(45)
    if path is not None:
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(*zip(ball, (1, -GATE_HALF)))
        ax.plot(*zip(ball, (1, GATE_HALF)))
        ax.scatter(*ball, marker='*', color='g')
        ax.scatter(*enemies[inside].T, color='r')
        ax.scatter(*enemies[~inside].T, color='b')
        ax.plot(*zip(*[(-1, -GOAL_AREA_HALF), (-GOAL_AREA_X, -GOAL_AREA_HALF), (-GOAL_AREA_X, GOAL_AREA_HALF), (-1, GOAL_AREA_HALF)]), c='b')
        ax.plot(*zip(*[(1, -GOAL_AREA_HALF), (GOAL_AREA_X, -GOAL_AREA_HALF), (GOAL_AREA_X, GOAL_AREA_HALF), (1, GOAL_AREA_HALF)]), c='r')
        if np.any(inside):
            r = max(np.linalg.norm(np.array([1, y]) - ball) for y in (-GATE_HALF, GATE_HALF))
            areas = [[ball + r * np.array([np.cos(angle), np.sin(angle)]) for angle in angles] for angles in [(_lower, lower), (upper, _upper)]]
            for p1, p2 in areas:
                ax.fill(*zip(ball, p1, p2), alpha=0.3)
        # ax.add_artist(plt.Circle(ball, SHOT_RANGE, fill=False))
        ax.set_xlim([-1, 1])
        ax.set_ylim([-Y_HALF, Y_HALF])
        ax.invert_yaxis()
        ax.set_title(str(value))
        ax.set_aspect('equal')
        fig.tight_layout()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        logging.info(path)
        plt.close(fig)
    return value


class Tracker(object):
    def __init__(self, observation):
        self.traceback = self.before = self.after = types.SimpleNamespace(observation=observation, owner=Owner(observation['ball_owned_team']))
        self.trajectory = []

    def __call__(self, observation, **kwargs):
        if self.before.owner in {Owner.me, Owner.enemy}:
            self.reset()
        self.before = self.after
        self.after = types.SimpleNamespace(observation=observation, owner=Owner(observation['ball_owned_team']))
        for key, value in kwargs.items():
            setattr(self.after, key, value)
        if Mode(observation['game_mode']) != Mode.normal or np.linalg.norm(self.after.observation['ball'][:2] - self.before.observation['ball'][:2]) > 0.3:
            self.reset()
        self.trajectory.append(self.after)

    def reset(self):
        self.traceback = self.before
        self.trajectory = []
