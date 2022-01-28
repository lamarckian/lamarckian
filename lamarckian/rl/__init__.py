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
import inspect
import types
import itertools
import contextlib
import functools
import hashlib
import pickle
import operator
import numbers
import random
import codecs
import collections.abc
import logging
import asyncio

import torch
import glom
import tqdm
import toolz
import ray

import lamarckian
from lamarckian.mdp import rollout
from . import agent, remote, skip, record, wrap


def to_device(device, *args, **kwargs):
    if args and kwargs:
        raise TypeError
    if args:
        tensors = []
        for value in args:
            if torch.is_tensor(value):
                tensors.append(value.to(device))
            elif isinstance(value, collections.abc.Sequence):
                tensors.append([t.to(device) for t in value])
            else:
                raise TypeError
    elif kwargs:
        tensors = {}
        for key, value in kwargs.items():
            if torch.is_tensor(value):
                tensors[key] = value.to(device)
            elif isinstance(value, collections.abc.Sequence):
                tensors[key] = [t.to(device) for t in value]
            else:
                raise TypeError(key, value)
    else: raise TypeError
    return tensors


def cat(tensors, *args, **kwargs):
    tensor = tensors[0]
    if torch.is_tensor(tensor):
        return torch.cat(tensors, *args, **kwargs)
    elif isinstance(tensor, collections.abc.Sequence):
        return [torch.cat(tensors, *args, **kwargs) for tensors in zip(*tensors)]
    else:
        raise TypeError(tensor)


def make_batch(get, batch_size, dim=0):
    tensors = get()
    batch = [tensors.values()]
    size = next(filter(torch.is_tensor, batch[-1])).shape[dim]
    while size < batch_size:
        tensors = get()
        batch.append(tensors.values())
        size += next(filter(torch.is_tensor, batch[-1])).shape[dim]
    return {key: cat(values, dim) for key, values in zip(tensors.keys(), zip(*batch))}


class Agents(dict):
    def __init__(self, agents):
        for id, agent in agents.items():
            self[id] = agent

    def close(self):
        for agent in self.values():
            agent.close()


class RL(lamarckian.evaluator.Evaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.me = glom.glom(kwargs['config'], 'rl.me', default=0)
        self.enemies = glom.glom(self.kwargs['config'], 'rl.enemies', default=[1])
        assert self.me not in self.enemies, (self.me, self.enemies)
        self.opponent_train = {}
        self.opponents_eval = [{}]
        self.opponents_eval_digest = ''

    def describe(self):
        return dict(blob=dict(me=self.me, enemies=self.enemies))

    def set_opponent_train(self, opponent):
        assert opponent
        self.opponent_train = opponent

    def get_opponent_train(self):
        return self.opponent_train

    def set_opponents_eval(self, opponents):
        assert isinstance(opponents, collections.abc.Iterable), type(opponents)
        assert opponents
        assert all(isinstance(opponent, dict) for opponent in opponents), [type(opponent) for opponent in opponents]
        self.opponents_eval = opponents
        if opponents[0]:
            self.opponents_eval_digest = hashlib.md5(pickle.dumps(opponents)).hexdigest()
        else:
            self.opponents_eval_digest = ''
        return self.opponents_eval_digest

    def get_opponents_eval(self):
        return self.opponents_eval

    def get_opponents_eval_digest(self):
        return self.opponents_eval_digest

    def reduce(self, *args, **kwargs):
        result = super().reduce(*args, **kwargs)
        digest = self.get_opponents_eval_digest()
        if digest:
            result['digest_opponents_eval'] = digest
        return result


def swap(size, opponent, random=random, **kwargs):
    if not opponent:
        return random.choice(list(range(size))), {}
    else:
        assert 1 + len(opponent) == size, (len(opponent), size)
        if len(opponent) == 1:
            if random.random() < glom.glom(kwargs['config'], 'rl.swap', default=0.5):
                (me,), (blob,) = opponent, opponent.values()
                enemy, = set(list(range(size))) - set(opponent)
                return me, {enemy: blob}
            else:
                me, = set(list(range(size))) - set(opponent)
                return me, opponent
        else:
            assert False, len(opponent)


class RL1(RL):
    """
    Base for all RL algorithms.

    :param mdp.create: The MDP class used to create the environment.
    :param rl.me: Configuration of the primary agent.
    :param rl.enemies: Configuration of other agents.
    :param config: Trainging configuration.
    """
    def __init__(self, state={}, **kwargs):
        super().__init__(state, **kwargs)
        cls = lamarckian.evaluator.parse(*glom.glom(kwargs['config'], 'mdp.create'), **kwargs)
        self.mdp = cls(**kwargs)
        self.hparam = lamarckian.util.Hparam()
        if hasattr(self.mdp, 'hparam'):
            self.hparam.__setstate__(self.mdp.hparam.__getstate__())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoding = self.describe()['blob']
        model = encoding['models'][self.me]
        module = lamarckian.evaluator.parse(*model['cls'], **kwargs)
        self._model = functools.partial(module, **model, **kwargs, reward=encoding['reward'])
        self.model = self._model()
        if state:
            self.set(state['decision'])
        self.model = self.model.to(self.device)
        self.model.train()
        self.models = {}
        for i in range(len(self.mdp)):
            model = encoding['models'][i]
            cls = lamarckian.evaluator.parse(*model['cls'], **kwargs)
            model = self.models[i] = cls(**model, **kwargs, reward=encoding['reward']).to(self.device)
            model.eval()
        self.agent = types.SimpleNamespace(**{key: lamarckian.evaluator.parse(*spec, **kwargs) for key, spec in encoding['agent'].items()})
        seed = glom.glom(kwargs['config'], 'seed', default=None)
        self.random = random.Random(seed)
        self.generator = torch.Generator(self.device)
        if seed is not None:
            self.generator.manual_seed(seed)

    def close(self):
        self.mdp.close()
        return super().close()

    def seed(self, seed):
        self.mdp.seed(seed)

    def initialize_blob(self):
        return lamarckian.model.to_blob(self._model().state_dict())

    def initialize(self):
        decision = dict(blob=self.initialize_blob())
        for coding, vector in self.hparam.initialize().items():
            decision[coding] = vector
        return decision

    def set_blob(self, blob):
        state_dict = self.model.state_dict()
        state_dict = lamarckian.model.from_blob(blob, state_dict.keys(), self.device)
        self.model.load_state_dict(state_dict)

    def set_hparam(self, state):
        self.hparam.__setstate__(state)

    def set(self, decision):
        self.set_blob(decision['blob'])
        self.hparam.set(decision)
        if hasattr(self.mdp, 'hparam'):
            self.mdp.hparam.__setstate__(self.hparam.__getstate__())

    def get_blob(self):
        return lamarckian.model.to_blob(self.model.state_dict())

    def get_hparam(self):
        return self.hparam.__getstate__()

    def get(self):
        decision = dict(blob=self.get_blob())
        for coding, vector in self.hparam.get().items():
            decision[coding] = vector
        return decision

    def spawn(self):
        opponent = self.get_opponent_train()
        return swap(len(self.models), opponent, random=self.random, **self.kwargs)

    def make_agents(self, blobs):
        agents = {}
        for index, blob in blobs.items():
            model = self.models[index]
            model.load_state_dict(lamarckian.model.from_blob(blob, model.state_dict().keys(), self.device))
            agents[index] = self.agent.eval(model)
        return Agents(agents)

    def evaluate1(self, seed, opponent, me=None, blob=None, **kwargs):
        if blob is not None:
            self.set_blob(blob)
        with torch.no_grad(), contextlib.closing(self.mdp.evaluating(seed)):
            try:
                self.model.eval()
                loop = asyncio.get_event_loop()
                if me is None:
                    me, opponent = swap(len(self.models), opponent, random=self.random, **self.kwargs)
                battle = self.mdp.reset(me, *opponent, loop=loop)
                with contextlib.closing(battle), contextlib.closing(self.agent.eval(self.model)) as agent, contextlib.closing(self.make_agents(opponent)) as agents:
                    costs = loop.run_until_complete(asyncio.gather(
                        rollout.get_cost(battle.controllers[0], agent),
                        *[rollout.get_cost(controller, agent) for controller, agent in zip(battle.controllers[1:], agents.values())],
                        *battle.ticks,
                    ))[:len(battle.controllers)]
                    result = battle.controllers[0].get_result()
                    if opponent:
                        result['digest_opponent_eval'] = hashlib.md5(pickle.dumps(opponent)).hexdigest()
                    return seed, max(costs), {**result, **kwargs}
            finally:
                self.model.train()

    def evaluate1_trajectory(self, seed, opponent, **kwargs):
        with torch.no_grad(), contextlib.closing(self.mdp.evaluating(seed)):
            try:
                self.model.eval()
                loop = asyncio.get_event_loop()
                me, opponent = swap(len(self.models), opponent, random=self.random, **self.kwargs)
                battle = self.mdp.reset(me, *opponent, loop=loop)
                with contextlib.closing(battle), contextlib.closing(self.agent.eval(self.model)) as agent, contextlib.closing(self.make_agents(opponent)) as agents:
                    data = loop.run_until_complete(asyncio.gather(
                        rollout.get_trajectory(battle.controllers[0], agent, **kwargs),
                        *[rollout.get_cost(controller, agent) for controller, agent in zip(battle.controllers[1:], agents.values())],
                        *battle.ticks,
                    ))[:len(battle.controllers)]
                    trajectory, exp = data[0]
                    result = battle.controllers[0].get_result()
                    if opponent:
                        result['digest_opponent_eval'] = hashlib.md5(pickle.dumps(opponent)).hexdigest()
                    return seed, sum(exp.get('cost', 1) for exp in trajectory), trajectory, exp, result
            finally:
                self.model.train()

    def evaluate(self):
        sample = glom.glom(self.kwargs['config'], 'sample.eval')
        opponents = self.get_opponents_eval()
        _, costs, results = zip(*[self.evaluate1(*args) for args in zip(range(sample), itertools.cycle(opponents))])
        self.cost += sum(costs)
        return results

    def iterate_trajectory(self, sample, **kwargs):
        opponents = self.get_opponents_eval()
        for args in zip(range(sample), itertools.cycle(opponents)):
            yield self.evaluate1_trajectory(*args, **kwargs)

    def __getstate__(self):
        state = super().__getstate__()
        state['mdp'] = self.mdp.__getstate__()
        return state

    @classmethod
    def remote_cls(cls, **kwargs):
        return kwargs.get('ray', ray).remote(**glom.glom(kwargs['config'], 'evaluator.ray'))(cls)


class Remote(RL):
    """
    Base class for all distributed RL algorithms.

    Actor is the class used to create (remote) works that are responsible for collecting rollouts.
    """
    Actor = None

    def __init__(self, state={}, **kwargs):
        super().__init__(state, **kwargs)
        try:
            torch.set_num_threads(glom.glom(kwargs['config'], 'rl.threads', default=1))
        except KeyError:
            logging.warning('learner use all threads')
        if (glom.glom(kwargs['config'], 'evaluator.progress', default=False) or 'index' not in kwargs) and 'ray' not in kwargs:
            progress = f"learner{kwargs.get('index', '')}"
        else:
            progress = None
        self.actors = self.create_actor(progress, **kwargs)
        self.rpc_all = lamarckian.util.rpc.All(self.actors, progress=progress, **kwargs)
        self.rpc_any = lamarckian.util.rpc.Any(self.actors, **kwargs)
        self.hparam = lamarckian.util.Hparam()
        self.hparam.__setstate__(self.rpc_any('get_hparam'))
        encoding = self.describe()['blob']
        model = encoding['models'][self.me]
        module = lamarckian.evaluator.parse(*model['cls'], **kwargs)
        self.model = module(**model, **kwargs, reward=encoding['reward'])
        if state:
            self.set(state['decision'])
        self.model.train()
        self.agent = types.SimpleNamespace(**{key: lamarckian.evaluator.parse(*spec, **kwargs) for key, spec in encoding['agent'].items()})

    def close(self):
        self.rpc_all.close()
        self.rpc_any.close()
        self.ray.get([actor.close.remote() for actor in self.actors])

    def create_actor(self, progress, **kwargs):
        """
        Create remote Actors that runs on different processes with Ray.
        """
        if 'ray' in kwargs:
            parallel = 1
        else:
            try:
                try:
                    parallel = kwargs['parallel']
                except KeyError:
                    parallel = glom.glom(kwargs['config'], 'evaluator.parallel')
            except KeyError:
                parallel = int(ray.cluster_resources()['CPU'] / glom.glom(kwargs['config'], 'evaluator.ray.num_cpus', default=1))
        assert parallel > 0, parallel
        cls = lamarckian.evaluator.parse(
            self.Actor, wrap.remote.training_switch,
            *glom.glom(kwargs['config'], 'evaluator.wrap', default=[]),
            **kwargs,
        )
        cls = cls.remote_cls(**kwargs)

        def create(index, **_kwargs):
            try:
                name = f"{kwargs['name']}/actor{index}"
            except KeyError:
                name = f"actor{index}"
            return cls.options(name=name, **_kwargs).remote(**{**kwargs, **dict(index=index, name=name)})

        size = glom.glom(kwargs['config'], 'evaluator.group', default=None)
        if 'index' in kwargs and isinstance(size, numbers.Integral):
            if size <= 0:
                sizes = [node['Resources'].get('CPU', 0) for node in self.ray.nodes()]
                sizes = [size for size in sizes if size > 0]
                size = int(min(sizes))
            groups = [(self.ray.util.placement_group([{'CPU': len(indexes)}]), indexes) for indexes in toolz.itertoolz.partition_all(size, range(parallel))]
            return [create(index, placement_group=placement_group) for placement_group, indexes in groups for index in indexes]
        else:
            return [create(index) for index in (range if progress is None else functools.partial(tqdm.trange, desc=progress))(parallel)]

    def seed(self, seed):
        name = inspect.getframeinfo(inspect.currentframe()).function
        self.rpc_all(name, seed)

    def __len__(self):
        return len(self.actors)

    def describe(self):
        self.sync_hparam(self.hparam.__getstate__())
        name = inspect.getframeinfo(inspect.currentframe()).function
        return self.rpc_any(name)

    def initialize(self):
        self.sync_hparam(self.hparam.__getstate__())
        name = inspect.getframeinfo(inspect.currentframe()).function
        return self.rpc_any(name)

    def sync_blob(self, blob):
        return self.rpc_all('set_blob', blob)

    def set_blob(self, blob):
        self.sync_blob(blob)
        state_dict = self.model.state_dict()
        state_dict = lamarckian.model.from_blob(blob, state_dict.keys(), next(iter(state_dict.values())).device)
        return self.model.load_state_dict(state_dict)

    def sync_hparam(self, state):
        return self.rpc_all('set_hparam', state)

    def set_hparam(self, state):
        self.sync_hparam(state)
        self.hparam.__setstate__(state)

    def set(self, decision):
        name = inspect.getframeinfo(inspect.currentframe()).function
        self.rpc_all(name, decision)
        state_dict = self.model.state_dict()
        state_dict = lamarckian.model.from_blob(decision['blob'], state_dict.keys(), next(iter(state_dict.values())).device)
        self.model.load_state_dict(state_dict)
        self.hparam.set(decision)

    def set_opponent_train(self, blobs):
        name = inspect.getframeinfo(inspect.currentframe()).function
        return self.rpc_all(name, blobs)

    def get_blob(self):
        return lamarckian.model.to_blob(self.model.state_dict())

    def get_hparam(self):
        return self.hparam.__getstate__()

    def get(self):
        decision = dict(blob=self.get_blob())
        for coding, vector in self.hparam.get().items():
            decision[coding] = vector
        return decision

    def training(self):
        self.sync_blob(self.get_blob())
        self.sync_hparam(self.hparam.__getstate__())
        var = {}
        with codecs.open(os.path.join(os.path.dirname(os.path.abspath(lamarckian.__file__)), 'import.py'), 'r', 'utf-8') as f:
            exec(f.read(), var)
        self.optimizer = eval('lambda params, lr: ' + glom.glom(self.kwargs['config'], 'train.optimizer'), var)(filter(lambda p: p.requires_grad, self.model.parameters()), self.hparam['lr'])
        training = super().training()
        self.rpc_all('training_on')

        def close():
            self.rpc_all('training_off')
            training.close()
        return types.SimpleNamespace(close=close)

    def evaluate(self):
        self.sync_blob(self.get_blob())
        opponents = self.get_opponents_eval()
        sample = glom.glom(self.kwargs['config'], 'sample.eval', default=len(self))
        items = self.rpc_any.map((('evaluate1', args, {}) for args in zip(range(sample), itertools.cycle(opponents))))
        costs, results = zip(*[items[1:] for items in sorted(items, key=operator.itemgetter(0))])
        self.cost += sum(costs)
        return results

    def iterate_trajectory(self, sample, **kwargs):
        self.sync_blob(self.get_blob())
        opponents = self.get_opponents_eval()
        return self.rpc_any.map((('evaluate1_trajectory', args, kwargs) for args in zip(range(sample), itertools.cycle(opponents))))

    def describe_rpc_all(self):
        return repr(self.rpc_all)

    def __getstate__(self):
        name = inspect.getframeinfo(inspect.currentframe()).function
        state = super().__getstate__()
        for key, value in self.rpc_any(name).items():
            if key not in state:
                state[key] = value
        return state

    @classmethod
    def remote_cls(cls, **kwargs):
        return kwargs.get('ray', ray).remote(**glom.glom(kwargs['config'], 'rl.ray'))(cls)


class Truncator(object):
    def __init__(self, rl, step, cast=rollout.cast):
        self.rl = rl
        self.step = step
        self.cast = cast
        self.done = True
        self.agent = types.SimpleNamespace(close=lambda: None)
        self.agents = {}
        self.battle = types.SimpleNamespace(close=lambda: None)
        self.tasks = {}
        self.loop = asyncio.new_event_loop()
        logging.getLogger('asyncio').setLevel(logging.CRITICAL)  # avoid asyncio warnings

    def close(self):
        self.agent.close()
        for agent in self.agents.values():
            agent.close()
        self.battle.close()
        for task in self.tasks:
            task.cancel()
        self.loop.stop()
        self.loop.close()

    def _reset(self):
        assert self.done
        self.done = False
        self.close()
        self.loop = asyncio.new_event_loop()
        me, self.opponent = self.rl.spawn()
        self.agent = self.rl.agent.train(self.rl.model, hparam=self.rl.hparam, generator=self.rl.generator, **self.rl.kwargs)
        self.agents = self.rl.make_agents(self.opponent)
        self.battle = self.rl.mdp.reset(me, *self.opponent, loop=self.loop)
        self.tasks = {asyncio.Task(rollout.get_cost(controller, agent), loop=self.loop) for controller, agent in zip(self.battle.controllers[1:], self.agents.values())} | {asyncio.Task(tick, loop=self.loop) for tick in self.battle.ticks}

    def __next__(self):
        if self.done:
            self._reset()
        task = asyncio.Task(rollout.get_trajectory(self.battle.controllers[0], self.agent, step=self.step, cast=self.cast), loop=self.loop)
        lamarckian.mdp.util.wait(task, self.tasks, self.loop)
        trajectory, exp = task.result()
        self.done = trajectory[-1]['done']
        # if self.done:
        #     self.loop.run_until_complete(asyncio.gather(*self.tasks))
        return trajectory, exp


def cumulate(reward, discount, terminal=0):
    credit = [terminal]
    for r in reversed(reward):
        credit.insert(0, r + discount * credit[0])
    return credit[:-1]
