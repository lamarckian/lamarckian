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
import types
import itertools
import operator

import torch.distributed
import glom
import tqdm
import port_for
import ray

import lamarckian
from . import hook


def parallel(rl):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    class RL(rl):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            setattr(self, PATH_FUNC, sum(self.rpc_all.fetch_all('__len__')))

        def __len__(self):
            return getattr(self, PATH_FUNC)
    return RL


def training_switch(rl):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    class RL(rl):
        def training_on(self):
            assert not hasattr(self, PATH_FUNC)
            setattr(self, PATH_FUNC, self.training())

        def training_off(self):
            getattr(self, PATH_FUNC).close()
            delattr(self, PATH_FUNC)
    return RL


def ddp(learner):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    def decorate(rl):
        def get_parallel(parallel, **kwargs):
            try:
                return glom.glom(kwargs['config'], 'evaluator.parallel')
            except KeyError:
                return int(ray.cluster_resources()['CPU'] / glom.glom(kwargs['config'], 'evaluator.ray.num_cpus', default=1)) // parallel

        class RL(rl):
            @lamarckian.evaluator.wrap.fix_ray
            @lamarckian.util.rpc.wrap.all
            @lamarckian.util.rpc.wrap.any
            class _Learner(learner):
                def setup_ddp(self, mode, **kwargs):
                    if mode == 'actor':
                        return self.actors
                    elif mode == 'address':
                        port = port_for.select_random(glom.glom(self.kwargs['config'], 'rpc.ports', default=None))
                        return f"tcp://{ray.util.get_node_ip_address()}:{port}"
                    else:
                        device = next(self.model.parameters()).device
                        assert device.type == 'cuda', device
                        torch.distributed.init_process_group(**kwargs, **glom.glom(self.kwargs['config'], 'ddp.init', default={}))
                        device = torch.device(f"cuda:{kwargs['rank']}")
                        model = self.model.to(device)
                        self.model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[kwargs['rank']], **glom.glom(self.kwargs['config'], 'ddp.model', default={}))

                def __call__(self):
                    cost = self.cost
                    outcome = super().__call__()
                    return self.cost - cost, outcome

            class Learner(_Learner):
                def __init__(self, *args, **kwargs):
                    _ = os.environ.pop('CUDA_VISIBLE_DEVICES')
                    super().__init__(*args, **{**kwargs, **dict(device=f"cuda:{kwargs['index']}")})
                    assert not hasattr(self, PATH_FUNC)
                    setattr(self, PATH_FUNC, _)

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, PATH_FUNC)
                cls = self.Learner.remote_cls(**kwargs)

                def create(index, parallel):
                    try:
                        name = f"{kwargs['name']}/learner{index}"
                    except KeyError:
                        name = f"learner{index}"
                    return cls.options(name=name).remote(*args, **{**kwargs, **dict(index=index, parallel=parallel, name=name)})
                parallel = 1 if 'ray' in kwargs else glom.glom(kwargs['config'], 'rl.parallel', default=torch.cuda.device_count() if torch.cuda.is_available() else 1)
                assert parallel > 0, parallel
                self.learners = [create(index=index, parallel=get_parallel(parallel, **kwargs)) for index in range(parallel)]
                self.actors = list(itertools.chain(*self.ray.get([learner.setup_ddp.remote('actor') for learner in self.learners])))
                setattr(self, PATH_FUNC, types.SimpleNamespace(
                    learner=types.SimpleNamespace(all=lamarckian.util.rpc.all.Flat(self.learners, **kwargs), any=lamarckian.util.rpc.Any(self.learners, **kwargs)),
                    actor=types.SimpleNamespace(all=lamarckian.util.rpc.All(self.actors, **kwargs), any=lamarckian.util.rpc.Any(self.actors, **kwargs)),
                ))
                if 'ray' not in kwargs and len(self.learners) > 1:
                    address = self.ray.get(self.learners[0].setup_ddp.remote('address'))
                    backend = glom.glom(kwargs['config'], 'ddp.backend', default='nccl')
                    self.ray.get([learner.setup_ddp.remote(
                        None,
                        backend=backend, init_method=address,
                        rank=i, world_size=parallel,
                    ) for i, learner in enumerate(tqdm.tqdm(self.learners, desc=f"DDP connect to {address} with {backend}"))])
                self.encoding = self.describe()

            def close(self):
                attr = getattr(self, PATH_FUNC)
                for _ in attr.__dict__.values():
                    for rpc in _.__dict__.values():
                        rpc.close()
                self.ray.get([learner.close.remote() for learner in attr.learners])

            def __len__(self):
                return len(self.actors)

            def describe(self):
                name = inspect.getframeinfo(inspect.currentframe()).function
                return getattr(self, PATH_FUNC).actor.any(name)

            def set(self, decision):
                name = inspect.getframeinfo(inspect.currentframe()).function
                return getattr(self, PATH_FUNC).learner.all(name, decision)

            def set_blob(self, blob):
                name = inspect.getframeinfo(inspect.currentframe()).function
                return getattr(self, PATH_FUNC).learner.all(name, blob)

            def get(self):
                name = inspect.getframeinfo(inspect.currentframe()).function
                return getattr(self, PATH_FUNC).learner.any(name)

            def get_blob(self):
                name = inspect.getframeinfo(inspect.currentframe()).function
                return getattr(self, PATH_FUNC).learner.any(name)

            def training(self):
                attr = getattr(self, PATH_FUNC)
                attr.learner.all('training_on')

                def close():
                    attr.learner.all('training_off')
                return types.SimpleNamespace(close=close)

            def __next__(self):
                return zip(*getattr(self, PATH_FUNC).learner.all.fetch_all('__call__'))

            def __call__(self):
                costs, outcomes = next(self)
                self.cost += sum(costs)
                return outcomes[0]

            def evaluate(self):
                attr = getattr(self, PATH_FUNC)
                attr.actor.all('set_blob', self.get_blob())
                opponents = self.get_opponents_eval()
                sample = glom.glom(self.kwargs['config'], 'sample.eval', default=len(self))
                items = attr.actor.any.map((('evaluate1', args, {}) for args in zip(range(sample), itertools.cycle(opponents))))
                costs, results = zip(*[items[1:] for items in sorted(items, key=operator.itemgetter(0))])
                self.cost += sum(costs)
                return results

            def iterate_trajectory(self, sample, **kwargs):
                attr = getattr(self, PATH_FUNC)
                attr.actor.all('set_blob', self.get_blob())
                opponents = self.get_opponents_eval()
                return attr.actor.any.map((('evaluate1_trajectory', args, kwargs) for args in zip(range(sample), itertools.cycle(opponents))))

            def set_opponent_train(self, blobs):
                name = inspect.getframeinfo(inspect.currentframe()).function
                return getattr(self, PATH_FUNC).actor.all(name, blobs)
        return RL
    return decorate
