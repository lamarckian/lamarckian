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

import os
import inspect
import contextlib
import itertools
import functools
import hashlib
import pickle
import random
import asyncio

import numpy as np
import torch

import lamarckian
from lamarckian.mdp import rollout
import lamarckian.mdp.gym as _module
import lamarckian.mdp.pong as module
from .agent import Rule as Agent

ROOT = os.path.dirname(__file__)


def test_random_process():
    sample = 2
    with contextlib.closing(module.MDP()) as mdp:
        for me in range(len(mdp)):
            encoding = mdp.describe()
            encoding_blob, = (encoding[key] for key in ('blob',))
            assert 'models' in encoding_blob
            mdp.seed(100)
            rs = mdp.get_random_state()
            # PvE
            agent = Agent(mdp, me)
            digests = (hashlib.md5(pickle.dumps(random.getstate())).hexdigest(), hashlib.md5(pickle.dumps(np.random.get_state())).hexdigest(), hashlib.md5(torch.random.get_rng_state().numpy().tostring()).hexdigest())
            _, results = lamarckian.mdp.evaluate(mdp, me, agent, itertools.repeat({}, sample))
            result = mdp.reduce(results)
            _, _results = lamarckian.mdp.evaluate(mdp, me, agent, itertools.repeat({}, sample))
            _result = mdp.reduce(_results)
            assert result['fitness'] == _result['fitness'], (result, _result)
            assert hashlib.md5(pickle.dumps(rs)).hexdigest() == hashlib.md5(pickle.dumps(mdp.get_random_state())).hexdigest()
            assert (hashlib.md5(pickle.dumps(random.getstate())).hexdigest(), hashlib.md5(pickle.dumps(np.random.get_state())).hexdigest(), hashlib.md5(torch.random.get_rng_state().numpy().tostring()).hexdigest()) == digests
            # PvP
            opponent = {enemy: Agent(mdp, enemy) for enemy in set(range(len(mdp))) - {me}}
            digests = (hashlib.md5(pickle.dumps(random.getstate())).hexdigest(), hashlib.md5(pickle.dumps(np.random.get_state())).hexdigest(), hashlib.md5(torch.random.get_rng_state().numpy().tostring()).hexdigest())
            _, results = lamarckian.mdp.evaluate(mdp, me, agent, itertools.repeat(opponent, sample))
            result = mdp.reduce(results)
            _, _results = lamarckian.mdp.evaluate(mdp, me, agent, itertools.repeat(opponent, sample))
            _result = mdp.reduce(_results)
            assert result['fitness'] == _result['fitness'], (result, _result)
            assert hashlib.md5(pickle.dumps(rs)).hexdigest() == hashlib.md5(pickle.dumps(mdp.get_random_state())).hexdigest()
            assert (hashlib.md5(pickle.dumps(random.getstate())).hexdigest(), hashlib.md5(pickle.dumps(np.random.get_state())).hexdigest(), hashlib.md5(torch.random.get_rng_state().numpy().tostring()).hexdigest()) == digests


def test_multi_objective():
    config = {}
    for path in [
        os.path.dirname(lamarckian.__file__) + '.yml',
        os.path.join(ROOT, 'wrap', 'behavior', 'weight.yml'), os.path.join(ROOT, 'wrap', 'behavior', 'weight_.yml'),
    ]:
        config = lamarckian.util.config.read(path, config)
    kwargs = dict(config=config)
    with contextlib.closing(functools.reduce(lambda x, wrap: wrap(x, **kwargs) if inspect.getfullargspec(wrap).varkw else wrap(x), map(lamarckian.util.parse.instance, config['mdp']['create']))(config=config)) as mdp:
        for me in range(len(mdp)):
            encoding = mdp.describe()
            assert len(encoding) == 2
            encoding_blob, encoding_real = (encoding[key] for key in ('blob', 'real'))
            model = _module.model.pong.fc.Module(**encoding_blob['models'][me]['kwargs'])
            model.eval()
            assert len(set(encoding_real['header'])) == len(encoding_real['header']) == len(encoding_real['boundary']) == 4, encoding_real['header']
            lower, upper = np.stack(list(encoding_real.values())).T
            decision = mdp.initialize()
            weight = decision['real']
            assert np.all(lower <= weight) and np.all(weight <= upper)
            mdp.set(decision)
            np.testing.assert_array_almost_equal(mdp.get()['real'], weight)
            # PvE
            agent = Agent(mdp, me)
            digests = (hashlib.md5(pickle.dumps(random.getstate())).hexdigest(), hashlib.md5(pickle.dumps(np.random.get_state())).hexdigest(), hashlib.md5(torch.random.get_rng_state().numpy().tostring()).hexdigest())
            with contextlib.closing(mdp.evaluating(0)):
                loop = asyncio.get_event_loop()
                battle = mdp.reset(me)
                with contextlib.closing(battle):
                    controller, = battle.controllers
                    trajectory, exp = loop.run_until_complete(asyncio.gather(
                        rollout.get_trajectory(controller, agent),
                        *battle.ticks,
                    ))[0]
                    result = controller.get_result()
                    assert len(result['objective']) == 2, result
            for exp in trajectory:
                assert len(exp['state']['inputs'][0].shape) == 1, exp['state']['inputs'][0].shape
            lamarckian.rl.cumulate(torch.Tensor([exp['reward'] for exp in trajectory]), 0.99)
            assert (hashlib.md5(pickle.dumps(random.getstate())).hexdigest(), hashlib.md5(pickle.dumps(np.random.get_state())).hexdigest(), hashlib.md5(torch.random.get_rng_state().numpy().tostring()).hexdigest()) == digests
            # PvP
            opponent = {enemy: Agent(mdp, enemy) for enemy in set(range(len(mdp))) - {me}}
            digests = (hashlib.md5(pickle.dumps(random.getstate())).hexdigest(), hashlib.md5(pickle.dumps(np.random.get_state())).hexdigest(), hashlib.md5(torch.random.get_rng_state().numpy().tostring()).hexdigest())
            loop = asyncio.get_event_loop()
            battle = mdp.reset(me, *opponent, loop=loop)
            with contextlib.closing(battle):
                trajectory, exp = loop.run_until_complete(asyncio.gather(
                    rollout.get_trajectory(battle.controllers[0], agent),
                    *[rollout.get_cost(controller, agent) for controller, agent in zip(battle.controllers[1:], opponent.values())],
                    *battle.ticks,
                ))[0]
                result = battle.controllers[0].get_result()
            assert len(result['objective']) == 2, result
            lamarckian.rl.cumulate(torch.Tensor([exp['reward'] for exp in trajectory]), 0.99)
            assert (hashlib.md5(pickle.dumps(random.getstate())).hexdigest(), hashlib.md5(pickle.dumps(np.random.get_state())).hexdigest(), hashlib.md5(torch.random.get_rng_state().numpy().tostring()).hexdigest()) == digests


def test_image():
    kwargs = dict()
    mdp = functools.reduce(lambda x, wrap: wrap(x, **kwargs) if inspect.getfullargspec(wrap).varkw else wrap(x), [
        module.MDP,
        _module.wrap.image.color.grey,
        _module.wrap.image.crop(34, -16, 16, -16), _module.wrap.image.downsample(2, 2), _module.wrap.image.resize(40, 32),
        _module.wrap.cast(), _module.wrap.image.color.scale(64, 255),
        _module.wrap.expand_dims(-1), _module.wrap.image.stack(4),
    ])
    with contextlib.closing(mdp()) as mdp:
        for me in range(len(mdp)):
            encoding = mdp.describe()
            model = _module.model.pong.conv.Module(**encoding['blob']['models'][me]['kwargs'])
            model.eval()
            digests = (hashlib.md5(pickle.dumps(random.getstate())).hexdigest(), hashlib.md5(pickle.dumps(np.random.get_state())).hexdigest(), hashlib.md5(torch.random.get_rng_state().numpy().tostring()).hexdigest())
            loop = asyncio.get_event_loop()
            battle = mdp.reset(me)
            with contextlib.closing(battle):
                controller, = battle.controllers
                trajectory, exp = loop.run_until_complete(asyncio.gather(
                    rollout.get_trajectory(controller, lamarckian.rl.pg.agent.Eval(model)),
                    *battle.ticks,
                ))[0]
            for exp in trajectory:
                assert len(exp['state']['inputs'][0].shape) == 3, exp['state']['inputs'][0].shape
            assert (hashlib.md5(pickle.dumps(random.getstate())).hexdigest(), hashlib.md5(pickle.dumps(np.random.get_state())).hexdigest(), hashlib.md5(torch.random.get_rng_state().numpy().tostring()).hexdigest()) == digests


def test_reward_blob():
    config = {}
    for path in [
        os.path.dirname(lamarckian.__file__) + '.yml',
        os.path.join(ROOT, 'wrap', 'behavior', 'blob.yml'),
    ]:
        config = lamarckian.util.config.read(path, config)
    kwargs = dict(config=config)
    with contextlib.closing(functools.reduce(lambda x, wrap: wrap(x, **kwargs) if inspect.getfullargspec(wrap).varkw else wrap(x), map(lamarckian.util.parse.instance, config['mdp']['create']))(config=config)) as mdp:
        for me in range(len(mdp)):
            encoding = mdp.describe()
            assert len(encoding) == 2
            encoding_blob, encoding_reward_blob = (encoding[key] for key in ('blob', 'reward_blob'))
            model = _module.model.pong.fc.Module(**encoding_blob['models'][me]['kwargs'])
            model.eval()
            assert 'model' in encoding_reward_blob
            decision = mdp.initialize()
            mdp.set(decision)
            # PvE
            agent = Agent(mdp, me)
            digests = (hashlib.md5(pickle.dumps(random.getstate())).hexdigest(), hashlib.md5(pickle.dumps(np.random.get_state())).hexdigest(), hashlib.md5(torch.random.get_rng_state().numpy().tostring()).hexdigest())
            with contextlib.closing(mdp.evaluating(0)):
                loop = asyncio.get_event_loop()
                battle = mdp.reset(me)
                with contextlib.closing(battle):
                    controller, = battle.controllers
                    trajectory, exp = loop.run_until_complete(asyncio.gather(
                        rollout.get_trajectory(controller, agent),
                        *battle.ticks,
                    ))[0]
                    result = controller.get_result()
                    assert len(result['objective']) == 2, result
            for exp in trajectory:
                assert len(exp['state']['inputs'][0].shape) == 1, exp['state']['inputs'][0].shape
            lamarckian.rl.cumulate(torch.Tensor([exp['reward'] for exp in trajectory]), 0.99)
            assert (hashlib.md5(pickle.dumps(random.getstate())).hexdigest(), hashlib.md5(pickle.dumps(np.random.get_state())).hexdigest(), hashlib.md5(torch.random.get_rng_state().numpy().tostring()).hexdigest()) == digests
            # PvP
            opponent = {enemy: Agent(mdp, enemy) for enemy in set(range(len(mdp))) - {me}}
            digests = (hashlib.md5(pickle.dumps(random.getstate())).hexdigest(), hashlib.md5(pickle.dumps(np.random.get_state())).hexdigest(), hashlib.md5(torch.random.get_rng_state().numpy().tostring()).hexdigest())
            loop = asyncio.get_event_loop()
            battle = mdp.reset(me, *opponent, loop=loop)
            with contextlib.closing(battle):
                trajectory, exp = loop.run_until_complete(asyncio.gather(
                    rollout.get_trajectory(battle.controllers[0], agent),
                    *[rollout.get_cost(controller, agent) for controller, agent in zip(battle.controllers[1:], opponent.values())],
                    *battle.ticks,
                ))[0]
                result = battle.controllers[0].get_result()
            assert len(result['objective']) == 2, result
            lamarckian.rl.cumulate(torch.Tensor([exp['reward'] for exp in trajectory]), 0.99)
            assert (hashlib.md5(pickle.dumps(random.getstate())).hexdigest(), hashlib.md5(pickle.dumps(np.random.get_state())).hexdigest(), hashlib.md5(torch.random.get_rng_state().numpy().tostring()).hexdigest()) == digests
