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

import random

import numpy as np
import torch
import glom

from lamarckian.rl.agent import Agent

EPS = float(np.finfo(np.float32).eps)


class Eval(Agent):
    def mask_legal(self, q, legal=None):
        if legal is not None:
            q = q * legal
        return q

    def __call__(self, state):
        with torch.no_grad():
            inputs = self.get_inputs(state)
            q, = self.forward(*inputs)
            try:
                legal = torch.FloatTensor(state['legal']).unsqueeze(0).to(self.device)
            except KeyError:
                legal = None
            q = self.mask_legal(q, legal)
            _, action = q.max(-1)
            exp = dict(
                inputs=inputs,
                q=q,
                action=action,
            )
            if legal is not None:
                exp['legal'] = legal
            return exp


class Train(Eval):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.random = random.Random(glom.glom(kwargs['config'], 'seed', default=None))

    def explore(self, q):
        if self.random.random() < self.kwargs['hparam']['epsilon']:
            return torch.randint(len(q), [1]).to(self.device)
        else:
            _, action = q.max(-1)
            return action

    def __call__(self, state):
        inputs = self.get_inputs(state)
        q, = self.forward(*inputs)
        try:
            legal = torch.FloatTensor(state['legal']).unsqueeze(0).to(self.device)
        except KeyError:
            legal = None
        q = self.mask_legal(q, legal)
        action = self.explore(q)
        exp = dict(
            inputs=inputs,
            q=q,
            action=action,
        )
        if legal is not None:
            exp['legal'] = legal
        return exp
