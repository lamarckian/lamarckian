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

import types

import numpy as np
import mujoco_py.utils
import glom

import lamarckian
from lamarckian.mdp import MDP as _MDP


class MDP(_MDP):
    class Controller(_MDP.Controller):
        def __init__(self, mdp, me=0):
            super().__init__(mdp)
            self.me = me
            self.sim = mdp.sim
            self.fitness = 0

        def get_state(self):
            return dict(inputs=self.mdp.get_inputs())

        async def __call__(self, **kwargs):
            sim = self.sim
            sim.data.ctrl[:] = kwargs['continuous'].squeeze(0).numpy()
            sim.step()
            self.reward = 0
            exp = dict(
                done=self.mdp.is_done(),
            )
            return exp

        def get_reward(self):
            return np.array([self.mdp.hparam['reward.score'] * self.reward])

        def get_result(self):
            return dict(
                fitness=self.fitness,
                objective=[],
            )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        path = glom.glom(kwargs['config'], 'mdp.mujoco.model')
        path = path.replace('MUJOCO_PATH', mujoco_py.utils.discover_mujoco())
        self.sim = mujoco_py.MjSim(mujoco_py.load_model_from_path(path))
        inputs = glom.glom(kwargs['config'], 'mdp.mujoco.inputs')
        self.get_inputs = lambda: [np.concatenate([getattr(self.sim.data, key) for key in inputs]).astype(np.float32)]  # [getattr(self.sim.data, key).astype(np.float32) for key in inputs]
        self.frame = 0
        frame = glom.glom(kwargs['config'], 'mdp.mujoco.frame')
        self.is_done = lambda: self.frame > frame
        self.hparam = lamarckian.util.Hparam()
        self.hparam.setup('reward.score', glom.glom(kwargs['config'], 'mdp.gym.reward.score', default=1), np.float)

    def close(self):
        self.sim.close()

    def describe_blob(self):
        encoding = dict(
            models=[dict(
                inputs=[dict(shape=a.shape) for a in self.get_inputs()],
                continuous=np.stack([np.zeros(self.sim.data.ctrl.shape), np.ones(self.sim.data.ctrl.shape)], -1),
            ) for _ in range(len(self))],
            reward=['score'],
        )
        return encoding

    def describe(self):
        return dict(blob=self.describe_blob())

    def initialize(self):
        return {}

    def set(self, *args, **kwargs):
        return {}

    def get(self):
        return {}

    def __len__(self):
        return 1

    def reset(self, me, loop=None):
        assert me == 0, me
        self.frame = 0
        self.sim.reset()
        return types.SimpleNamespace(
            controllers=[self.Controller(self)],
            ticks=[],
            close=lambda: None,
        )

    def render(self, *args, **kwargs):
        return self.sim.render(*args, **kwargs)
