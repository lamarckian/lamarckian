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

import math

import numpy as np
import torch
import torch.nn.functional as F

from lamarckian.rl.agent import Agent

EPS = float(np.finfo(np.float32).eps)


class Eval(Agent):
    def get_discrete_probs(self, logits, legal=None):
        if legal is not None:
            mask = torch.zeros_like(legal)
            mask[legal == 0] = -math.inf
            logits = logits + mask
        return F.softmax(logits, -1)

    def get_discrete_dist(self, *args, **kwargs):
        probs = self.get_discrete_probs(*args, **kwargs)
        return torch.distributions.Categorical(probs)

    def get_continuous_dist(self, output):
        return torch.distributions.Normal(*output)

    def __call__(self, state):
        with torch.no_grad():
            inputs = self.get_inputs(state)
            exp = {**self.forward(*inputs), **dict(inputs=inputs)}
            if 'legal' in state:
                exp['legal'] = [torch.FloatTensor(legal).unsqueeze(0).to(self.device) for legal in state['legal']]
                probs = [self.get_discrete_probs(logits, legal) for logits, legal in zip(exp['discrete'], exp['legal'])]
            else:
                probs = [self.get_discrete_probs(logits) for logits in exp['discrete']]
            exp['discrete'] = [p.max(-1)[1] for p in probs]
            if 'continuous' in exp:
                exp['continuous_dist'] = self.get_continuous_dist(exp['continuous'])
                exp['continuous'] = exp['continuous_dist'].mean
            return exp


class Train(Eval):
    def get_discrete_probs(self, *args, **kwargs):
        probs = super().get_discrete_probs(*args, **kwargs)
        prob_min = self.kwargs['hparam']['prob_min']
        if prob_min > 0:
            probs = (probs + prob_min) / (1 + prob_min * probs.shape[-1])
        return probs

    def explore(self, dist):
        return torch.multinomial(dist.probs, 1, True, generator=self.kwargs['generator'])[0]

    def __call__(self, state):
        inputs = self.get_inputs(state)
        exp = {**self.forward(*inputs), **dict(inputs=inputs)}
        if 'legal' in state:
            exp['legal'] = [torch.FloatTensor(legal).unsqueeze(0).to(self.device) for legal in state['legal']]
            assert len(exp['discrete']) == len(exp['legal']), (len(exp['discrete']), len(exp['legal']))
            exp['discrete_dist'] = [self.get_discrete_dist(logits, legal) for logits, legal in zip(exp['discrete'], exp['legal'])]
        else:
            exp['discrete_dist'] = [self.get_discrete_dist(logits) for logits in exp['discrete']]
        exp['discrete'] = list(map(self.explore, exp['discrete_dist']))
        if 'continuous' in exp:
            exp['continuous_dist'] = self.get_continuous_dist(exp['continuous'])
            exp['continuous'] = exp['continuous_dist'].sample()
        return exp
