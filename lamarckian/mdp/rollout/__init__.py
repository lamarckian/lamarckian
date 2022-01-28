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

import traceback

import numpy as np


async def cast(controller, agent, state):
    exp = agent(state)
    exp['state'] = state
    exp.update(await controller(**exp))
    state = controller.get_state()
    exp['reward'] = controller.get_reward()
    return state, exp


async def get_trajectory(controller, agent, step=np.iinfo(np.int).max, message=False, cast=cast):
    assert step > 0, step
    state = controller.get_state()
    trajectory = []
    try:
        for _ in range(step):
            if message:
                msg = repr(controller)
                state, exp = await cast(controller, agent, state)
                exp['message'] = msg
            else:
                state, exp = await cast(controller, agent, state)
            trajectory.append(exp)
            if exp['done']:
                break
        exp = dict(
            state=state,
            inputs=agent.get_inputs(state),
        )
        if message:
            exp['message'] = repr(controller)
        return trajectory, exp
    except:
        traceback.print_exc()
        raise


async def get_cost(controller, agent, step=np.iinfo(np.int).max):
    assert step > 0, step
    state = controller.get_state()
    cost = 0
    try:
        for _ in range(step):
            exp = agent(state)
            exp.update(await controller(**exp))
            state = controller.get_state()
            cost += exp.get('cost', 1)
            if exp['done']:
                break
        return cost
    except:
        traceback.print_exc()
        raise
