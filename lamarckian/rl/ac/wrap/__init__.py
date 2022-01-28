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

import inspect

import numpy as np


def fix_skip(rl):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'
    from lamarckian.rl.ac import cumulate, disassemble

    def pad(skip, *args):
        tensors = []
        for trajectory in args:
            assert len(trajectory) == len(skip), (len(trajectory), len(skip))
            padded = []
            for exp, n in zip(trajectory, skip):
                padded.append(exp)
                padded += [np.zeros_like(exp)] * n
            tensors.append(np.stack(padded))
        return tensors

    def restore(skip, *args):
        tensors = []
        for padded in args:
            assert len(padded) == len(skip) + skip.sum().item(), (len(padded), len(skip) + skip.sum().item())
            trajectory = []
            for n in skip:
                trajectory.append(padded[0])
                padded = padded[n + 1:]
            assert len(trajectory) == len(skip), (len(trajectory), len(skip))
            tensors.append(np.stack(trajectory))
        return tensors

    class RL(rl):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            encoding = self.describe()['blob']
            setattr(self, PATH_FUNC, np.array([self.hparam[f"discount_{name}"] for name in encoding['reward']]))

        def rollout(self):
            trajectory, exp, results = super().rollout()
            discount = np.expand_dims(np.logical_not([exp['done'] for exp in trajectory]), -1) * getattr(self, PATH_FUNC)
            terminal = np.zeros_like(getattr(self, PATH_FUNC)) if exp is None else self.model(*exp['inputs'])['critic'].squeeze(0).cpu().numpy()
            skip = np.array([exp.get('cost', 1) - 1 for exp in trajectory])
            assert np.all(skip >= 0)
            reward = np.array([exp['reward'] for exp in trajectory])
            _reward, _discount = pad(
                skip,
                reward,
                discount,
            )
            _credit = cumulate(_reward, _discount, terminal)
            credit, = restore(skip, _credit)
            reward = np.array(disassemble(credit, discount, terminal))
            # np.testing.assert_almost_equal(np.array(cumulate(reward, discount, terminal)), credit)
            for exp, r in zip(trajectory, reward):
                exp['reward'] = r
            return trajectory, exp, results
    return RL
