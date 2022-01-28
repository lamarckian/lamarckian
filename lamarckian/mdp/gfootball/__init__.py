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
import numbers
import itertools
import enum
import random
import traceback
import logging
import asyncio

import numpy as np
import torch
import glom
import deepmerge
import gfootball
import cv2

import lamarckian
from .. import MDP as _MDP
from . import wrap, model, wrapper, rule, util, record
from .util import Owner, Mode, Action


PRETTY_ACTION = dict(left='←', top_left='↖', top='↑', top_right='↗', right='→', bottom_right='↘', bottom='↓', bottom_left='↙')


class PvE(_MDP):
    class Evaluating(object):
        def __init__(self, mdp, seed):
            assert not hasattr(mdp, '_evaluating')
            mdp._evaluating = self
            self.mdp = mdp
            self.rs = random.getstate()
            try:
                self.state = mdp.env.get_state()
            except:
                traceback.print_exc()
            self.mdp.seed(seed)

        def close(self):
            delattr(self.mdp, '_evaluating')
            random.setstate(self.rs)
            try:
                self.mdp.env.set_state(self.state)
            except:
                traceback.print_exc()

    class Controller(_MDP.Controller):
        def __init__(self, mdp, me, inputs):
            super().__init__(mdp)
            self.me = me
            self.inputs = inputs

        def get_state(self):
            exp = dict(inputs=self.inputs)
            try:
                exp['legal'] = self.mdp.env.unwrapped.legal[self.me]
            except AttributeError:
                pass
            return exp

        async def __call__(self, **kwargs):
            self.action, = map(lambda x: x.item(), kwargs['discrete'])
            (self.inputs,), self.reward, done, self.mdp.info = self.mdp.env.step(self.mdp.encode_action[self.action])
            self.mdp.frame += 1
            self.mdp.done = done or self.mdp.frame >= self.mdp.length
            return dict(done=self.mdp.done)

        def get_reward(self):
            return np.array([self.mdp.hparam['reward.score'] * self.reward])

        def get_result(self):
            assert self.mdp.done
            observation = self.mdp.env.unwrapped.observation()[self.me]
            score_me, score_enemy = observation['score']
            return dict(
                fitness=score_me - score_enemy,
                win=1 if score_me > score_enemy else 0 if score_me < score_enemy else 0.5,
                score_me=score_me,
                score_enemy=score_enemy,
                yellow=observation['left_team_yellow_card'].sum(),
                red=np.sum(~observation['left_team_active']),
                objective=[],
            )

    def add_macro_action_set(self, encode_action):
        index = sum(1 for key in encode_action if isinstance(key, numbers.Number) and key >= 0)
        try:
            for i, name in enumerate(self.env.unwrapped._macro_action_set):
                assert isinstance(name, str), name
                encode_action[name] = name
                encode_action[index + i] = name
        except AttributeError:
            pass
        return encode_action

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for old, new in glom.glom(kwargs['config'], 'mdp.gfootball.override', default={}).items():
            module, name = old.rsplit('.', 1)
            new = lamarckian.util.parse.instance(new)
            new.__name__ = name
            setattr(lamarckian.util.parse.attr(module), name, new)
        self.env = self.create_env()
        _env = self.env.unwrapped
        action_set = {str(action): i for i, action in enumerate(_env._env._action_set)}
        self.encode_action = {i: i for name, i in action_set.items() if name != 'builtin_ai'}
        self.encode_action['builtin_ai'] = self.encode_action[-1] = action_set['builtin_ai']
        self.add_macro_action_set(self.encode_action)
        self.Action = enum.Enum('Action', zip([str(_env._env._action_set[value]) if isinstance(value, numbers.Number) else value for key, value in self.encode_action.items() if isinstance(key, numbers.Number) and key >= 0], itertools.count()))
        self.states = {}
        self.length = glom.glom(kwargs['config'], 'mdp.length', default=np.iinfo(np.int).max)
        self.hparam = lamarckian.util.Hparam()
        self.hparam.setup('reward.score', glom.glom(kwargs['config'], 'mdp.gfootball.reward.score', default=1), np.float)

    def create_env(self, **kwargs):
        func = getattr(gfootball.env, glom.glom(self.kwargs['config'], 'mdp.gfootball.env', default='create_environment'))
        kwargs = deepmerge.always_merger.merge(kwargs, dict(
            env_name='11_vs_11_hard_stochastic',
            representation='simple115v2',
            other_config_options=dict(action_set='v2'),
            number_of_right_players_agent_controls=1,
            logdir=os.path.expanduser(os.path.expandvars(self.kwargs['config']['root'])),
        ))
        kwargs = deepmerge.always_merger.merge(kwargs, glom.glom(self.kwargs['config'], 'mdp.gfootball.create'))
        _env = env = func(**kwargs)
        for s in glom.glom(self.kwargs['config'], 'mdp.gfootball.wrapper', default=[]):
            env = lamarckian.util.parse.instance(s, env=env, kwargs=self.kwargs)
            _env._wrappers_with_support.add(env.__class__.__name__)
        return env

    def close(self):
        return self.env.close()

    def get_action_names(self):
        header = [action.name for action in self.Action]
        for i, name in enumerate(header):
            if name in PRETTY_ACTION:
                header[i] = PRETTY_ACTION[name]
        return header

    def describe_blob(self):
        names = self.get_action_names()
        return dict(
            models=[dict(
                inputs=[dict(shape=tuple(x if isinstance(x, numbers.Number) else len(x) for x in header), header=header) for header in self.env.unwrapped.header_state],
                discrete=[names],
            ) for _ in range(len(self))],
            reward=['score'],
        )

    def describe(self):
        return dict(blob=self.describe_blob())

    def evaluating(self, seed):
        deterministic = glom.glom(self.kwargs['config'], 'mdp.gfootball.deterministic', default=0)
        if deterministic >= 0:
            return self.Evaluating(self, seed + deterministic)
        else:
            return types.SimpleNamespace(close=lambda: None)

    def seed(self, seed):
        path = glom.glom(self.kwargs['config'], 'mdp.gfootball.state', default='')
        if path:
            self.state = torch.load(os.path.expandvars(os.path.expanduser(path)))
            logging.info(f'load gfootball state from {path}')
        else:
            try:
                if seed not in self.states:
                    env = self.create_env(other_config_options=dict(game_engine_random_seed=seed))
                    self.states[seed] = env.get_state()
                    env.close()
                self.env.set_state(self.states[seed])
                random.seed(seed)
                self.state = dict(env=self.states[seed], random=random.getstate())
            except:
                traceback.print_exc()

    def render(self, *args, **kwargs):
        image = self.env.render(*args, **kwargs)
        if isinstance(image, np.ndarray):
            if glom.glom(self.kwargs['config'], 'mdp.gfootball.render.subtitle', default=True):
                height, width, _ = image.shape
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                thickness = 6
                for controller in self.controllers:
                    if hasattr(controller, 'action'):
                        text = controller.action if isinstance(controller.action, str) else Action(controller.action).name
                        if self.env.unwrapped._agent._action is not None:
                            action = self.env.unwrapped._agent._action[controller.me]
                            try:
                                name = Action(action).name
                                if name != text:
                                    text += f" -> {name}"
                            except ValueError:
                                pass
                        (_width, _height), baseline = cv2.getTextSize(text, font, scale, thickness)
                        cv2.putText(image, text, (width - _width if controller.me else 0, _height + baseline), font, scale, (255, 255, 255), thickness)
                        cv2.putText(image, text, (width - _width if controller.me else 0, _height + baseline), font, scale, (0, 0, 0), thickness // 2)
                cv2.putText(image, '\t'.join([str(self.frame)] + [text for text in (repr(controller) for controller in self.controllers) if not text.startswith('<')]), (0, height), font, scale, (255, 255, 255))
            if glom.glom(self.kwargs['config'], 'mdp.gfootball.render.rgb', default=True):
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def __len__(self):
        return 1

    def reset(self, me, loop=None):
        self.done = False
        self.frame = 0
        inputs, = self.env.reset()
        self.controllers = [self.Controller(self, me, inputs)]
        self.__setstate__(self.__dict__.pop('state', None))
        return types.SimpleNamespace(
            controllers=self.controllers,
            ticks=[],
            close=lambda: None,
        )

    def __getstate__(self):
        return dict(env=self.env.get_state(), random=random.getstate())

    def __setstate__(self, state):
        if state is not None:
            self.env.set_state(state['env'])
            random.setstate(state['random'])


class PvP(PvE):
    class Controller(PvE.Controller):
        def __init__(self, mdp, me, inputs, enemy, loop=None):
            super().__init__(mdp, me, inputs)
            self.enemy = enemy
            self.queue = types.SimpleNamespace(
                action=asyncio.Queue(loop=loop),
                exp=asyncio.Queue(loop=loop),
            )

        async def __call__(self, **kwargs):
            self.action, = map(lambda x: x.item(), kwargs['discrete'])
            await self.queue.action.put(self.mdp.encode_action[self.action])
            return await self.queue.exp.get()

    def __len__(self):
        return 2

    def reset(self, *args, loop=None):
        self.done = False
        self.frame = 0
        states = self.env.reset()
        self.controllers = [self.Controller(self, me, inputs, (me + 1) % 2, loop=loop) for me, inputs in enumerate(states)]
        self.__setstate__(self.__dict__.pop('state', None))
        return types.SimpleNamespace(
            controllers=[self.controllers[me] for me in args],
            ticks=[self.rule(self.controllers[me]) for me in set(range(len(self))) - set(args)] + [self.tick(self.controllers)],
            close=lambda: None,
        )

    async def tick(self, controllers):
        assert not self.done
        try:
            while True:
                actions = [await controller.queue.action.get() for controller in controllers]
                states, rewards, self.done, self.info = self.env.step(actions)
                self.frame += 1
                self.done = self.done or self.frame >= self.length
                for controller, inputs, reward in zip(controllers, states, rewards):
                    controller.inputs = inputs
                    controller.reward = reward
                    await controller.queue.exp.put(dict(done=self.done))
                if self.done:
                    break
        except GeneratorExit:
            pass
        except:
            traceback.print_exc()
            raise

    async def rule(self, controller):
        try:
            while True:
                action = np.array('builtin_ai')
                exp = await controller(discrete=[action])
                if exp['done']:
                    break
        except GeneratorExit:
            pass
        except:
            traceback.print_exc()
            raise
