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

import types
import copy
import traceback
import asyncio

import numpy as np
import slimevolleygym
import glom

import lamarckian
from lamarckian.mdp import MDP as _MDP


class MDP(_MDP):
    class Evaluating(object):
        def __init__(self, problem, seed):
            self.problem = problem
            self.env = problem.env
            self.np_random = copy.deepcopy(self.env.unwrapped.game.np_random)
            self.env.seed(seed)

        def close(self):
            self.env.unwrapped.game.np_random = self.np_random

    class Controller(_MDP.Controller):
        def __init__(self, mdp, state, me=0):
            super().__init__(mdp)
            self.state = state
            self.me = me
            self.env = mdp.env
            self.fitness = 0

        def get_state(self):
            return dict(inputs=[self.state])

        async def __call__(self, **kwargs):
            action, = map(int, kwargs['discrete'])
            self.state, self.reward, self.done, info = self.env.step(self.mdp.action_table[action])
            self.fitness += self.reward
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
        self.env = slimevolleygym.SlimeVolleyEnv()
        self.action_table = [
            [0, 0, 0],  # NOOP
            [1, 0, 0],  # LEFT (forward)
            [1, 0, 1],  # UPLEFT (forward jump)
            [0, 0, 1],  # UP (jump)
            [0, 1, 1],  # UPRIGHT (backward jump)
            [0, 1, 0],  # RIGHT (backward)
        ]
        self.hparam = lamarckian.util.Hparam()
        self.hparam.setup('reward.score', glom.glom(kwargs['config'], 'mdp.gym.reward.score', default=1), np.float)

    def close(self):
        self.env.close()

    def describe_blob(self):
        names = [
            'NOOP',
            'LEFT',
            'UPLEFT',
            'UP',
            'UPRIGHT',
            'RIGHT',
        ]
        encoding = dict(
            models=[dict(
                inputs=[dict(shape=self.env.observation_space.shape)],
                discrete=[names],
            ) for _ in range(len(self))],
            reward=['score'],
        )
        for model in encoding['models']:
            try:
                model['inputs'][0]['header'] = ['x y vx vy ball_x ball_y ball_vx ball_vy op_x op_y op_vx op_vy'.split()]
            except AttributeError:
                pass
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

    # def evaluating(self, seed):
    #     return self.Evaluating(self, seed)

    def __len__(self):
        return 1

    def reset(self, me, loop=None):
        assert me == 0, me
        state = self.env.reset()
        return types.SimpleNamespace(
            controllers=[self.Controller(self, state)],
            ticks=[],
            close=lambda: None,
        )

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)


class MDP2(MDP):
    class Controller(MDP.Controller):
        def __init__(self, mdp, state, me, loop=None):
            super().__init__(mdp, state, me)
            self.queue = types.SimpleNamespace(
                action=asyncio.Queue(loop=loop),
                exp=asyncio.Queue(loop=loop),
            )

        async def __call__(self, **kwargs):
            action, = map(int, kwargs['discrete'])
            await self.queue.action.put(action)
            exp = await self.queue.exp.get()
            self.reward = exp.pop('reward')
            return exp

    def __len__(self):
        return 2

    def reset(self, *args, loop=None):
        state = self.env.reset()
        controllers = [self.Controller(self, state, me, loop=loop) for me in args]
        return types.SimpleNamespace(
            controllers=controllers,
            ticks=[self.tick(controllers)],
            close=lambda: None,
        )

    async def tick(self, controllers):
        try:
            while True:
                actions = [await controller.queue.action.get() for controller in controllers]
                actions = [self.action_table[action] for action in actions]
                state, reward, done, info = self.env.step(*actions)
                states = [state, info['otherObs']]
                rewards = [reward, -reward]
                for controller, state, reward in zip(controllers, states, rewards):
                    controller.state = state
                    await controller.queue.exp.put(dict(reward=reward, done=done))
                if done:
                    break
        except GeneratorExit:
            pass
        except:
            traceback.print_exc()
            raise
