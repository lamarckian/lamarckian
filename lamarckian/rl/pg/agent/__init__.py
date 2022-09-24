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

import numpy as np
import torch

from lamarckian.rl import Categorical
from lamarckian.rl.agent import Agent

EPS = float(np.finfo(np.float32).eps)


class Eval(Agent):
    def get_continuous_dist(self, output):
        return torch.distributions.Normal(*output)

    def __call__(self, state):
        with torch.no_grad():
            inputs = self.get_inputs(state)
            exp = {**self.forward(*inputs), **dict(inputs=inputs)}
            if 'legal' in state:
                exp['legal'] = [torch.from_numpy(legal).unsqueeze(0).to(self.device) for legal in state['legal']]
                exp['discrete_dist'] = [Categorical(logits, legal) for logits, legal in zip(exp['discrete'], exp['legal'])]
            else:
                exp['discrete_dist'] = [Categorical(logits) for logits in exp['discrete']]
            exp['discrete'] = [dist.probs.max(-1)[1] for dist in exp['discrete_dist']]
            if 'continuous' in exp:
                exp['continuous_dist'] = self.get_continuous_dist(exp['continuous'])
                exp['continuous'] = exp['continuous_dist'].mean
            return exp


class Train(Eval):
    def explore(self, dist):
        shape = dist.probs.shape
        probs = dist.probs.view(-1, shape[-1])
        return torch.multinomial(probs, 1, True, generator=self.kwargs['generator']).view(*shape[:-1])

    def __call__(self, state):
        inputs = self.get_inputs(state)
        exp = {**self.forward(*inputs), **dict(inputs=inputs)}
        if 'legal' in state:
            exp['legal'] = [torch.from_numpy(legal).unsqueeze(0).to(self.device) for legal in state['legal']]
            assert len(exp['discrete']) == len(exp['legal']), (len(exp['discrete']), len(exp['legal']))
            exp['discrete_dist'] = [Categorical(logits, legal) for logits, legal in zip(exp['discrete'], exp['legal'])]
        else:
            exp['discrete_dist'] = [Categorical(logits) for logits in exp['discrete']]
        exp['discrete'] = list(map(self.explore, exp['discrete_dist']))
        if 'continuous' in exp:
            exp['continuous_dist'] = self.get_continuous_dist(exp['continuous'])
            exp['continuous'] = exp['continuous_dist'].sample()
        return exp
