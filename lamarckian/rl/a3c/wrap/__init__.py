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

import inspect
import types

import numpy as np
import torch

import lamarckian


def actor(actor):
    class Actor(actor):
        def __init__(self, *args, **kwargs):
            torch.set_num_threads(1)
            super().__init__(*args, **kwargs)

        def gradient(self, blob):
            self.set_blob(blob)
            cost = self.cost
            self.optimizer.zero_grad()
            outcome = self.backward()
            outcome['gradient'] = [param.grad.cpu().numpy() for param in self.model.parameters()]
            return self.cost - cost, outcome
    return Actor


def learner(learner):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    class Learner(learner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            setattr(self, PATH_FUNC, lamarckian.util.rpc.wrap.any_count(lamarckian.util.rpc.Any)(self.actors, **kwargs))

        def close(self):
            getattr(self, PATH_FUNC).close()
            return super().close()

        def training(self):
            training = super().training()
            blob = self.get_blob()
            rpc = getattr(self, PATH_FUNC)
            for _ in range(len(self.actors)):
                rpc.send('gradient', blob)

            def close():
                for _ in range(len(rpc)):
                    rpc.receive()
                return training.close()
            return types.SimpleNamespace(close=close)

        def __call__(self):
            rpc = getattr(self, PATH_FUNC)
            cost, outcome = rpc.receive()
            rpc.send('gradient', self.get_blob())
            self.cost += cost
            self.optimizer.zero_grad()
            for param, grad in zip(self.model.parameters(), outcome['gradient']):
                param.grad = torch.from_numpy(np.array(grad))
            self.optimizer.step()
            return outcome
    return Learner
