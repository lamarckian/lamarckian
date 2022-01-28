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
import types
import itertools
import contextlib
import enum
import logging
import traceback
import asyncio

import numpy as np
import glom

import lamarckian

Action = enum.Enum('Action', zip('up stop down'.split(), itertools.count()))

with contextlib.redirect_stdout(None):
    from .pong_env import PongSinglePlayerEnv, PongDoublePlayerEnv
from .. import MDP as _MDP
from . import agent, wrap


class MDP(_MDP):
    class Evaluating(object):
        def __init__(self, mdp, seed):
            assert not hasattr(mdp, '_evaluating')
            mdp._evaluating = self
            self.mdp = mdp
            self.rs = mdp.get_random_state()
            self.mdp.seed(seed)

        def close(self):
            delattr(self.mdp, '_evaluating')
            self.mdp.random.setstate(self.rs)

    class Controller(_MDP.Controller):
        def __init__(self, mdp, me, enemy, state, loop=None):
            super().__init__(mdp)
            self.me = me
            self.enemy = enemy
            self.state = state
            self.queue = types.SimpleNamespace(
                action=asyncio.Queue(loop=loop),
                exp=asyncio.Queue(loop=loop),
            )

        def get_state(self):
            return dict(inputs=[self.state])

        def get_bat(self):
            return self.mdp.get_bats()[self.me]

        def get_bat_enemy(self):
            return self.mdp.get_bats()[self.enemy]

        def get_position_ball(self):
            arena = self.mdp.get_arena()
            width = arena.right - arena.left
            height = arena.bottom - arena.top
            ball = self.mdp.get_ball()
            return np.array([(ball._rect.x - arena.left) / width, (ball._rect.y - arena.top) / height])

        def get_position_bat(self):
            arena = self.mdp.get_arena()
            height = arena.bottom - arena.top
            return (self.get_bat()._rect.y - arena.top) / height

        def get_position_bat_enemy(self):
            arena = self.mdp.get_arena()
            height = arena.bottom - arena.top
            return (self.get_bat_enemy()._rect.y - arena.top) / height

        async def __call__(self, **kwargs):
            action, = map(int, kwargs['discrete'])
            await self.queue.action.put(action)
            exp = await self.queue.exp.get()
            self.reward = exp.pop('reward')
            return exp

        def get_reward(self):
            return np.array([self.mdp.hparam['reward.score'] * self.reward])

        def get_result(self):
            scores = self.mdp.get_scores()
            score_me, score_enemy = scores[self.me], scores[self.enemy]
            score_max = self.mdp.score_max
            return dict(
                fitness=((score_me - score_enemy) + score_max) / score_max / 2,
                score=score_me - score_enemy,
                score_me=score_me, score_enemy=score_enemy,
                win=1 if score_me > score_enemy else 0 if score_me < score_enemy else 0.5,
                objective=[],
            )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        os.environ['SDL_AUDIODRIVER'] = 'dsp'
        self.env = PongDoublePlayerEnv()
        game = self.env._game
        try:
            game._max_step_per_round = glom.glom(kwargs['config'], 'mdp.pong.max_step_per_round')
            logging.warning(f'change max_step_per_round to {game._max_step_per_round}')
        except KeyError:
            pass
        try:
            game._max_num_rounds = glom.glom(kwargs['config'], 'mdp.pong.max_num_rounds')
            logging.warning(f'change max_num_rounds to {game._max_num_rounds}')
        except KeyError:
            pass
        # self.length = (game._max_step_per_round + 1) * game._max_num_rounds
        self.length = glom.glom(kwargs['config'], 'mdp.length', default=np.iinfo(int).max)
        self.score_max = game._max_num_rounds
        ball = self.get_ball()
        self.random = ball.random
        try:
            ball_init = eval('lambda mdp: ' + glom.glom(kwargs['config'], 'mdp.pong.ball_init'))
            arena = self.get_arena()

            def wrap(func):
                def _(*args, **kwargs):
                    y = ball_init(self)
                    assert 0 <= y <= 1, y
                    ball._y_init = arena.top + y * (arena.bottom - arena.top)
                    return func(*args, **kwargs)
                return _
            ball.reset = wrap(ball.reset)
            logging.warning('change pong ball_init')
        except KeyError:
            pass
        self.hparam = lamarckian.util.Hparam()
        self.hparam.setup('reward.score', glom.glom(kwargs['config'], 'mdp.pong.reward.weight.score', default=1), np.float)

    def close(self):
        return self.env.close()

    def describe_blob(self):
        names = list(Action.__members__.keys())
        return dict(
            models=[dict(
                inputs=[dict(shape=space.shape)],
                discrete=[names],
            ) for space in self.env.observation_space.spaces],
            reward=['score'],
        )

    def describe(self):
        return dict(blob=self.describe_blob())

    def evaluating(self, seed):
        return self.Evaluating(self, seed)

    def seed(self, seed):
        self.random.seed(seed)

    def get_random_state(self):
        return self.random.getstate()

    def render(self):
        return self.env._render(mode=glom.glom(self.kwargs['config'], 'mdp.render.mode', default='rgb_array'))

    def __len__(self):
        return len(self.env.observation_space.spaces)

    def reset(self, *args, loop=None):
        self.frame = 0
        states = self.env._reset()
        controllers = [self.Controller(self, me, (me + 1) % 2, state, loop=loop) for me, state in enumerate(states)]
        return types.SimpleNamespace(
            controllers=[controllers[me] for me in args],
            ticks=[self.rule(controllers[me]) for me in set(range(len(self))) - set(args)] + [self.tick(controllers)],
            close=lambda: None,
        )

    async def tick(self, controllers):
        try:
            while True:
                actions = [await controller.queue.action.get() for controller in controllers]
                states, rewards, done, info = self.env._step(actions)
                done = done or self.frame >= self.length
                for controller, state, reward in zip(controllers, states, rewards):
                    controller.state = state
                    await controller.queue.exp.put(dict(reward=reward, done=done))
                self.frame += 1
                if done:
                    break
        except GeneratorExit:
            pass
        except:
            traceback.print_exc()
            raise

    async def rule(self, controller):
        try:
            while True:
                action = agent.rule(self, controller.me).value
                exp = await controller(discrete=[action])
                if exp['done']:
                    break
        except GeneratorExit:
            pass
        except:
            traceback.print_exc()
            raise

    def get_arena(self):
        return self.env._game._arena

    def get_ball(self):
        return self.env._game._ball

    def get_bats(self):
        game = self.env._game
        return [game._left_bat, game._right_bat]

    def get_scores(self):
        game = self.env._game
        return [game._score_left, game._score_right]
