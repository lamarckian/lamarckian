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

import torch.distributed
import glom
import tqdm
import port_for
import ray.services

import lamarckian
from . import hook


def parallel(rpc=lambda self: self.rpc_all):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    def decorate(rl):
        class RL(rl):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, PATH_FUNC)
                setattr(self, PATH_FUNC, sum(rpc(self).fetch_all('__len__')))

            def __len__(self):
                return getattr(self, PATH_FUNC)
        return RL
    return decorate


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
    @lamarckian.util.rpc.wrap.all
    @lamarckian.util.rpc.wrap.any
    @lamarckian.util.rpc.wrap.gather
    class Learner(learner):
        def setup_ddp(self, **kwargs):
            if kwargs:
                device = next(self.model.parameters()).device
                assert device.type == 'cuda', device
                for key, value in kwargs.pop('env').items():
                    os.environ[key] = str(value)
                torch.distributed.init_process_group(**kwargs)
                self.model = torch.nn.parallel.DistributedDataParallel(self.model)
            else:
                port = port_for.select_random(glom.glom(self.kwargs['config'], 'rpc.ports', default=None))
                return f"tcp://{ray.services.get_node_ip_address()}:{port}"

        def __call__(self):
            cost = self.cost
            outcome = super().__call__()
            return self.cost - cost, outcome

        def iterate_trajectory(self, *args, **kwargs):
            return list(super().iterate_trajectory(*args, **kwargs))

    def decorate(rl):
        def get_parallel(parallel, **kwargs):
            try:
                return glom.glom(kwargs['config'], 'evaluator.parallel')
            except KeyError:
                return int(ray.cluster_resources()['CPU'] / glom.glom(kwargs['config'], 'evaluator.ray.num_cpus', default=1)) // parallel

        @parallel()
        class RL(rl):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                cls = Learner
                cls = cls.remote_cls(**self.kwargs)

                def create(index, parallel):
                    try:
                        name = f"{kwargs['name']}/learner{index}"
                    except KeyError:
                        name = f"learner{index}"
                    return cls.options(name=name).remote(*args, **{**self.kwargs, **dict(index=index, parallel=parallel, name=name)})
                parallel = 1 if 'ray' in kwargs else glom.glom(kwargs['config'], 'rl.parallel', default=torch.cuda.device_count() if torch.cuda.is_available() else 1)
                assert parallel > 0, parallel
                self.learners = [create(index=index, parallel=get_parallel(parallel, **kwargs)) for index in range(parallel)]
                self.rpc_all = lamarckian.util.rpc.All(self.learners, **kwargs)
                self.rpc_any = lamarckian.util.rpc.Any(self.learners, **kwargs)
                if 'ray' not in kwargs and len(self.learners) > 1:
                    address = self.ray.get(self.learners[0].setup_ddp.remote())
                    backend = glom.glom(kwargs['config'], 'ddp.backend', default='gloo')
                    self.ray.get([learner.setup_ddp.remote(
                        env=glom.glom(kwargs['config'], 'ddp.env', default={}), backend=backend,
                        init_method=address,
                        rank=i, world_size=parallel,
                    ) for i, learner in enumerate(tqdm.tqdm(self.learners, desc=f"DDP connect to {address} with {backend}"))])

            def close(self):
                self.rpc_all.close()
                self.rpc_any.close()
                self.ray.get([learner.close.remote() for learner in self.learners])

            def describe(self):
                name = inspect.getframeinfo(inspect.currentframe()).function
                return self.rpc_any(name)

            def set(self, decision):
                name = inspect.getframeinfo(inspect.currentframe()).function
                return self.rpc_all(name, decision)

            def set_blob(self, blob):
                name = inspect.getframeinfo(inspect.currentframe()).function
                return self.rpc_all(name, blob)

            def set_opponent_train(self, blobs):
                name = inspect.getframeinfo(inspect.currentframe()).function
                return self.rpc_all(name, blobs)

            def get(self):
                name = inspect.getframeinfo(inspect.currentframe()).function
                return self.rpc_any(name)

            def get_blob(self):
                name = inspect.getframeinfo(inspect.currentframe()).function
                return self.rpc_any(name)

            def training(self):
                self.rpc_all('training_on')

                def close():
                    self.rpc_all('training_off')
                return types.SimpleNamespace(close=close)

            def __call__(self):
                costs, outcomes = zip(*self.rpc_all.fetch_all('__call__'))
                self.cost += sum(costs)
                return outcomes[0]

            def evaluate(self):
                name = inspect.getframeinfo(inspect.currentframe()).function
                return self.rpc_any(name)

            def iterate_trajectory(self, *args, **kwargs):
                name = inspect.getframeinfo(inspect.currentframe()).function
                return self.rpc_any(name, *args, **kwargs)
        return RL
    return decorate
