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
import math
import contextlib
import types
import ctypes
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
import glom
import zmq
import humanfriendly
import psutil
import filelock
import port_for

import lamarckian
from . import util, wrap, record
from .. import ac

NAME = os.path.basename(os.path.dirname(__file__))


@wrap.rpc.all
@lamarckian.util.rpc.wrap.any
class _Actor(ac.Actor):
    DIM = 1

    def __init__(self, *args, **kwargs):
        torch.set_num_threads(1)
        gather = kwargs.pop('gather')
        super().__init__(*args, **{**kwargs, **dict(device='cpu')})
        self.truncator.close()
        self.truncator = util.Truncator(self, self.hparam['truncation'])
        self.iteration = 0
        assert not hasattr(self, 'gather')
        self.gather = types.SimpleNamespace(running=True, training=threading.Event(), busy=True)
        self.gather.thread = threading.Thread(target=self.sending, args=(gather['url'], lamarckian.util.serialize.SERIALIZE[gather['serializer']]))
        self.gather.thread.start()

    def close(self):
        self.gather.running = False
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(self.gather.thread.ident),
            ctypes.py_object(SystemExit),
        )
        super().close()
        self.gather.thread.join()

    def training(self):
        training = super().training()
        self.gather.training.set()

        def close():
            self.gather.training.clear()
            while self.gather.busy:
                time.sleep(0.1)
            return training.close()
        return types.SimpleNamespace(close=close)

    def rollout(self):
        return next(self.truncator)

    def set_blob_(self, blob, iteration):
        self.set_blob(blob)
        self.iteration = iteration

    def get_iteration(self):
        return self.iteration

    def to_tensor(self, trajectory, exp):
        tensors = super().to_tensor(trajectory, exp)
        tensors = {key: [t.unsqueeze(1) for t in value] if isinstance(value, (tuple, list)) else value.unsqueeze(1) for key, value in tensors.items()}
        tensors['inputs'] = tuple(t.unsqueeze(1) for t in map(torch.cat, zip(*[exp['inputs'] for exp in trajectory + [exp]])))
        return tensors

    def sending(self, url, serialize):
        compact = glom.glom(self.kwargs['config'], 'rl.ppo.compact', default=1)
        assert compact > 0, compact
        context = zmq.Context()
        with torch.no_grad(), contextlib.closing(context.socket(zmq.REQ)) as socket:
            socket.connect(url)
            recv = socket.recv
            socket.recv = lambda: setattr(socket, 'recv', recv)
            while self.gather.running:
                self.gather.training.wait()
                self.gather.busy = True
                gather = util.Gather()
                for _ in range(compact):
                    trajectory, exp, results = self.rollout()
                    gather(sum(exp.get('cost', 1) for exp in trajectory), self.to_tensor(trajectory, exp), results)
                self.gather.busy = False
                tensors = {key: lamarckian.rl.cat([tensors[key] for tensors in gather.tensors], self.DIM) for key in gather.tensors[0]}
                socket.recv()
                socket.send(serialize((gather.cost, tensors, gather.results, self.get_iteration())))
            socket.recv()
        context.term()


class Actor(_Actor):  # ray's bug
    pass


class Learner(lamarckian.rl.Learner):
    Actor = Actor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparam.setup('lr', glom.glom(kwargs['config'], 'train.lr'), np.float)
        for key in 'policy critic entropy'.split():
            self.hparam.setup(f'weight_loss_{key}', glom.glom(kwargs['config'], f"rl.ac.weight_loss.{key}"), np.float)
        self.loss_critic = getattr(F, glom.glom(kwargs['config'], 'rl.ac.loss_critic', default='mse_loss'))
        self.iteration = 0
        self.hparam.setup('batch_size', glom.glom(kwargs['config'], 'train.batch_size'), np.int)
        self.hparam.setup('clip', glom.glom(kwargs['config'], f'rl.{NAME}.clip', default=np.finfo(np.float32).max), np.float)
        self.hparam.setup('reuse', glom.glom(kwargs['config'], f'rl.{NAME}.reuse', default=1), np.int)
        self.norm_advantage = eval('lambda advantage: ' + glom.glom(kwargs['config'], f'rl.ac.norm_advantage', default='advantage'))
        if kwargs.get('broadcaster', True):
            self.create_broadcaster(self.actors)
        else:
            self.broadcaster = util.broadcaster.Fake()

    def close(self):
        self.prefetcher.close()
        self.broadcaster.close()
        return super().close()

    def create_actors(self, parallel, *args, **kwargs):
        try:
            if 'ray' in self.kwargs or glom.glom(self.kwargs['config'], 'ray.local_mode', default=False) or not glom.glom(self.kwargs['config'], 'c.prefetcher', default=True):
                raise ImportError()
            from pylamarckian.rl.ppo import Prefetcher
        except ImportError:
            from .util import Prefetcher
        with filelock.FileLock(lamarckian.util.rpc.util.LOCK_PORT):
            port = port_for.select_random(glom.glom(self.kwargs['config'], 'rpc.ports', default=None))
            self.prefetcher = Prefetcher(
                f"tcp://*:{port}", 1, dim=self.Actor.DIM, device=self.device,
                parallel=min(glom.glom(self.kwargs['config'], 'rl.ppo.prefetcher.parallel', default=psutil.cpu_count()), parallel),
                serializer=glom.glom(self.kwargs['config'], 'rl.ppo.prefetcher.serializer', default=glom.glom(self.kwargs['config'], 'rpc.serializer')),
                capacity=glom.glom(self.kwargs['config'], 'rl.ppo.prefetcher.capacity', default=1),
            )
        host = self.ray.util.get_node_ip_address()
        gather = dict(url=f"tcp://{host}:{port}", serializer=self.prefetcher.serializer)
        return super().create_actors(parallel, *args, **kwargs, remote=lambda cls, *args, **kwargs: cls.remote(*args, **kwargs, gather=gather))

    def create_broadcaster(self, actors):
        try:
            if 'ray' in self.kwargs or not glom.glom(self.kwargs['config'], 'c.broadcaster', default=False):
                raise ImportError()
            from .util import broadcaster
            Broadcaster = getattr(broadcaster, glom.glom(self.kwargs['config'], 'rl.ppo.rpc_all', default=lamarckian.util.rpc.All.__name__))
            self.broadcaster = Broadcaster(actors, self.model, lambda: self.iteration, **self.kwargs)
        except ImportError:
            self.broadcaster = util.broadcaster.Async(actors, lambda: (self.get_blob(), self.iteration), **self.kwargs)

    def get_batch_size(self):
        if self.Actor.DIM:
            return math.ceil(self.hparam['batch_size'] / self.hparam['truncation'])
        else:
            return self.hparam['batch_size']

    def training(self):
        training = super().training()
        if hasattr(self.broadcaster, 'wrap_optimizer'):
            self.broadcaster.wrap_optimizer(self.optimizer)
        self.prefetcher.batch_size = self.get_batch_size()
        self.discount = torch.FloatTensor([self.hparam[f"discount_{name}"] for name in self.encoding['blob']['reward']]).to(self.device)
        self.gae = torch.FloatTensor([self.hparam[f"gae_{name}"] for name in self.encoding['blob']['reward']]).to(self.device)
        return training

    def forward(self, old):
        inputs = tuple(input.view(np.multiply.reduce(input.shape[:2]), *input.shape[2:]) for input in old['inputs'])
        outputs = self.model(*inputs)
        shape = old['inputs'][0].shape[:2]
        outputs = {key: value.view(*shape, *value.shape[1:]) if torch.is_tensor(value) else [t.view(*shape, *t.shape[1:]) for t in value] for key, value in outputs.items()}
        values = outputs['critic']
        outputs['critic'], outputs['critic_'] = values[:-1], values[1:]
        logp = []
        entropy = []
        with contextlib.closing(self.agent.train(self.model, hparam=self.hparam)) as agent:
            if outputs['discrete']:
                legal = old.get('legal', [None] * len(outputs['discrete']))
                assert len(legal) == len(outputs['discrete']), (len(legal), len(outputs['discrete']))
                for legal, logits, discrete in zip(legal, outputs['discrete'], torch.unbind(old['discrete'], -1)):
                    dist = lamarckian.rl.Categorical(logits[:-1], legal)
                    logp.append(dist.log_prob(discrete))
                    entropy.append(dist.entropy())
            if 'continuous' in outputs:
                dist = agent.get_continuous_dist([t[:-1] for t in outputs['continuous']])
                logp += torch.unbind(dist.log_prob(old['continuous']), -1)
                entropy += torch.unbind(dist.entropy(), -1)
        outputs['logp'] = logp = torch.stack(logp, -1)
        outputs['entropy'] = torch.stack(entropy, -1)
        outputs['ratio'] = torch.exp(logp - old['logp'])
        return outputs

    def estimate(self, old, new):
        with torch.no_grad():
            discount = torch.logical_not(old['done']).to(old['reward'].dtype) * self.discount
            advantage = torch.stack(ac.gae(old['reward'], new['critic'], new['critic_'], discount, self.gae))
            credit = new['critic'] + advantage
        return dict(advantage=advantage, credit=credit)

    def get_loss_policy(self, old, new):
        ratio = new['ratio'].mean(-1)
        advantage = self.norm_advantage(new['advantage']).sum(-1).to(ratio.dtype)
        return -torch.min(
            advantage * ratio,
            advantage * ratio.clamp(min=1 - self.hparam['clip'], max=1 + self.hparam['clip']),
        )

    def get_loss_critic(self, old, new):
        return self.loss_critic(new['critic'], new['credit'].to(new['critic'].dtype), reduction='none')

    def get_losses(self, old, new):
        policy = self.get_loss_policy(old, new)
        critic = self.get_loss_critic(old, new)
        assert len(policy.shape) == 1 or len(policy.shape) + 1 == len(critic.shape), (policy.shape, critic.shape)
        return dict(policy=policy.mean(), entropy=-new['entropy'].mean()), critic.view(-1, critic.shape[-1]).mean(0)

    def __next__(self):
        return self.prefetcher()

    def __call__(self):
        cost, old, results, _ = next(self)
        assert cost > 0, cost
        self.cost += cost
        for _ in range(self.hparam['reuse']):
            self.optimizer.zero_grad()
            new = self.forward(old)
            new.update(self.estimate(old, new))
            losses, loss_critic = self.get_losses(old, new)
            loss = sum(loss * self.hparam[f"weight_loss_{key}"] for key, loss in losses.items()) + loss_critic.sum() * self.hparam['weight_loss_critic']
            loss.backward()
            self.optimizer.step()
            self.iteration += 1
            self.broadcaster()
        return dict(results=results, loss=loss, losses=losses, loss_critic=loss_critic, ratio=new['ratio'])


@wrap.record.stale(NAME)
class _Evaluator(Learner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recorder = lamarckian.util.recorder.Recorder.new(**kwargs)
        self.saver = lamarckian.evaluator.Saver(self)
        self.profiler = lamarckian.evaluator.record.Profiler(self.cost, len(self), **kwargs)
        self.recorder.register(
            lamarckian.util.counter.Time(**glom.glom(kwargs['config'], 'record.save')),
            lambda *args, **kwargs: self.saver(),
        )
        self.recorder.register(
            lamarckian.util.counter.Time(**glom.glom(kwargs['config'], 'record.scalar')),
            lambda *args, **kwargs: self.profiler(self.cost),
        )
        self.recorder.register(
            lamarckian.rl.record.Rollout.counter(**kwargs),
            lambda *args, **kwargs: lamarckian.rl.record.Rollout.new(**kwargs)(self.cost, self.get_blob()),
        )
        self.recorder.register(
            lamarckian.util.counter.Time(**glom.glom(kwargs['config'], 'record.scalar')),
            lambda outcome: lamarckian.util.record.Scalar(self.cost, **{
                **{f'{NAME}/loss': outcome['loss'].item()},
                **{f'{NAME}/losses/{key}': loss.item() for key, loss in outcome['losses'].items()},
                **{f'{NAME}/losses/critic': outcome['loss_critic'].sum().item()},
                **{f'{NAME}/losses/critic_{key}': loss.item() for key, loss in zip(self.encoding['blob']['reward'], outcome['loss_critic'])},
                **self.results(),
                **{f'{NAME}/{key}': value for key, value in record.get_ratio(self, outcome).items()},
                **{f'{NAME}/broadcast/{key}': value for key, value in getattr(self.broadcaster, 'profile', {}).items()},
            }),
        )
        self.recorder.register(
            lamarckian.util.counter.Time(**glom.glom(kwargs['config'], 'record.model')),
            lambda outcome: lamarckian.rl.record.Model(f"{NAME}/model", self.cost, {key: value.cpu().numpy() for key, value in self.model.state_dict().items()}, glom.glom(kwargs['config'], 'record.model.sort', default='bytes')),
        )
        self.recorder.put(lamarckian.rl.record.Graph(f"{NAME}/graph", self.cost))
        self.recorder.put(lamarckian.util.record.Text(self.cost, **{f"{NAME}/model_size": humanfriendly.format_size(sum(value.cpu().numpy().nbytes for value in self.model.state_dict().values()))}))
        self.recorder.put(lamarckian.util.record.Text(self.cost, topology=repr(self.rpc_all), prefetcher=str(getattr(self.prefetcher, 'parallel', 1))))

    def close(self):
        self.saver.close()
        super().close()
        self.recorder.close()

    def training(self):
        self.results = lamarckian.rl.record.Results(**self.kwargs)
        return super().training()

    def __call__(self):
        outcome = super().__call__()
        self.results += outcome['results']
        self.recorder(outcome)
        return outcome


class Evaluator(_Evaluator):  # ray's bug
    pass
