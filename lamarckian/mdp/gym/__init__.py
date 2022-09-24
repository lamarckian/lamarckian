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

import logging
import copy
import types
import importlib
import traceback
import collections.abc
import asyncio

import numpy as np
import torch
import gym
import glom

import lamarckian
from lamarckian.mdp import MDP as _MDP
from . import wrap, model


def cast_env(env, t):
    while True:
        if isinstance(env, t):
            return env
        env = env.env


class MDP(_MDP):
    class Evaluating(object):
        def __init__(self, problem, seed):
            self.problem = problem
            self.env = problem.env
            try:
                self.np_random = copy.deepcopy(self.env.unwrapped.np_random)
            except AttributeError:
                pass
            self.env.seed(seed)

        def close(self):
            try:
                self.env.unwrapped.np_random = self.np_random
            except AttributeError:
                pass

    class Controller(_MDP.Controller):
        def __init__(self, mdp, state):
            super().__init__(mdp)
            self.state = state
            self.env = mdp.env
            self.fitness = 0

        def get_state(self):
            return dict(inputs=[self.state])

        async def __call__(self, **kwargs):
            try:
                action, = map(int, kwargs['discrete'])
            except ValueError:
                action = kwargs['continuous'].squeeze(0).numpy()
            self.state, self.reward, self.done, info = self.env.step(action)
            self.fitness += self.reward
            self.mdp.frame += 1
            exp = dict(
                done=self.done,
                info=info,
            )
            return exp

        def get_reward(self):
            return np.array([self.mdp.hparam['reward.score'] * self.reward])

        def get_result(self):
            return dict(
                fitness=self.fitness,
                objective=[],
            )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for module in glom.glom(kwargs['config'], 'mdp.gym.import', default=[]):
            importlib.import_module(module)
        self.env = gym.make(glom.glom(kwargs['config'], 'mdp.gym.env'))
        self.length = glom.glom(kwargs['config'], 'mdp.length', default=np.iinfo(int).max)
        if glom.glom(kwargs['config'], 'mdp.gym.unwrapped', default=False):
            self.env = self.env.unwrapped
        elif self.length < np.iinfo(int).max:
            try:
                _env = cast_env(self.env, gym.wrappers.TimeLimit)
                logging.warning(f'change length from {_env._max_episode_steps} to {self.length}')
                _env._max_episode_steps = self.length
            except:
                pass
        self.hparam = lamarckian.util.Hparam()
        self.hparam.setup('reward.score', glom.glom(kwargs['config'], 'mdp.gym.reward.score', default=1), np.float)

    def close(self):
        self.env.close()

    def describe_blob(self):
        encoding = dict(
            models=[dict(
                inputs=[dict(shape=self.env.observation_space.shape)],
            ) for _ in range(len(self))],
            reward=['score'],
        )
        for model in encoding['models']:
            try:
                model['inputs'][0]['header'] = [self.env.unwrapped.state_name]
            except AttributeError:
                pass
            if isinstance(self.env.action_space, gym.spaces.discrete.Discrete):
                try:
                    model['discrete'] = [self.env.unwrapped.get_action_meanings()]
                except AttributeError:
                    model['discrete'] = [list(map(str, range(self.env.action_space.n)))]
            else:
                model['continuous'] = np.stack([self.env.action_space.low, self.env.action_space.high], -1)
        return encoding

    def describe(self):
        return dict(blob=self.describe_blob())

    def initialize(self):
        return {}

    def set(self, *args, **kwargs):
        return {}

    def get(self):
        return {}

    def seed(self, seed):
        return self.env.seed(seed)

    def evaluating(self, seed):
        return self.Evaluating(self, seed)

    def get_random_state(self):
        return self.env.unwrapped.np_random

    def __len__(self):
        return 1

    def reset(self, me, loop=None):
        assert me == 0, me
        self.frame = 0
        state = self.env.reset()
        return types.SimpleNamespace(
            controllers=[self.Controller(self, state)],
            ticks=[],
            close=lambda: None,
        )

    def render(self):
        return self.env.render(mode=glom.glom(self.kwargs['config'], 'mdp.render.mode', default='rgb_array'))


class MA(MDP):
    class Controller(MDP.Controller):
        def __init__(self, mdp, state, me, enemy, loop=None):
            super().__init__(mdp, state)
            self.me = me
            self.enemy = enemy
            self.queue = types.SimpleNamespace(
                action=asyncio.Queue(loop=loop),
                exp=asyncio.Queue(loop=loop),
            )

        async def __call__(self, **kwargs):
            try:
                action, = map(int, kwargs['discrete'])
            except (KeyError, ValueError):
                action = kwargs['continuous'].squeeze(0).numpy()
            await self.queue.action.put(action)
            exp = await self.queue.exp.get()
            self.reward = exp.pop('reward')
            return exp

    def describe_blob(self):
        encoding = dict(
            models=[dict(
                inputs=[dict(shape=space.shape)],
            ) for space in self.env.observation_space.spaces],
            reward=['score'],
        )
        for model, space in zip(encoding['models'], self.env.action_space.spaces):
            if isinstance(space, gym.spaces.discrete.Discrete):
                model['discrete'] = [list(map(str, range(space.n)))]
            else:
                model['continuous'] = np.stack([space.low, space.high], -1)
        return encoding

    def __len__(self):
        return len(self.env.observation_space.spaces)

    def reset(self, *args, loop=None):
        self.frame = 0
        states = self.env.reset()
        controllers = [self.Controller(self, state, me, (me + 1) % len(self), loop=loop) for me, state in enumerate(states)]
        return types.SimpleNamespace(
            controllers=[controllers[me] for me in args],
            ticks=[self.rule(controllers[me]) for me in set(range(len(self))) - set(args)] + [self.tick(controllers)],
            close=lambda: None,
        )

    async def tick(self, controllers):
        while True:
            actions = [await controller.queue.action.get() for controller in controllers]
            states, rewards, done, info = self.env.step(actions)
            if isinstance(done, collections.abc.Sequence):
                done = done[0]
            done = done or self.frame >= self.length
            for controller, state, reward in zip(controllers, states, rewards):
                controller.state = state
                await controller.queue.exp.put(dict(reward=reward, done=done))
            self.frame += 1
            if done:
                break

    async def rule(self, controller):
        space = self.env.action_space.spaces[controller.me]
        while True:
            if isinstance(space, gym.spaces.discrete.Discrete):
                exp = await controller(discrete=[0])
            else:
                exp = await controller(continuous=torch.from_numpy((space.low + space.high) / 2).unsqueeze(0))
            if exp['done']:
                break
