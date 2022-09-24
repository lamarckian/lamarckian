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
import inspect
import functools
import multiprocessing

import glom
import zmq
import msgpack
import port_for
import ray

import lamarckian
from . import seed_torch


def seed_env(mdp):
    class MDP(mdp):
        def reset(self, *args, **kwargs):
            try:
                self.seed(glom.glom(self.kwargs['config'], 'seed'))
            except KeyError:
                pass
            return super().reset(*args, **kwargs)
    return MDP


def interact(mdp):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    from .interact_gui import run

    class MDP(mdp):
        class Controller(mdp.Controller):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, PATH_FUNC)
                setattr(self, PATH_FUNC, types.SimpleNamespace(frame=0))

            def get_state(self):
                attr = getattr(self.mdp, PATH_FUNC)
                state = super().get_state()
                if self is attr.controller:
                    _attr = getattr(self, PATH_FUNC)
                    _attr.state = state
                    if not _attr.frame:
                        attr.socket.send(msgpack.dumps(('reset', (state,), {}), default=lamarckian.util.serialize.encode))
                        attr.socket.recv()
                return state

            async def __call__(self, *args, **kwargs):
                attr = getattr(self.mdp, PATH_FUNC)
                if self is attr.controller:
                    _attr = getattr(self, PATH_FUNC)
                    attr.socket.send(msgpack.dumps(('__call__', (_attr.frame,), {key: value for key, value in kwargs.items() if key in {'discrete', 'continuous'}}), default=lamarckian.util.serialize.encode))
                    result = msgpack.loads(attr.socket.recv(), object_hook=lamarckian.util.serialize.decode, strict_map_key=False)
                    for key, value in result.items():
                        kwargs[key] = value
                    _attr.frame += 1
                exp = await super().__call__(*args, **kwargs)
                if self is attr.controller:
                    _attr = getattr(self, PATH_FUNC)
                    exp['state'] = _attr.state
                    for key, value in result.items():
                        exp[key] = value
                    _attr.exp = exp
                    if 'message' in exp:
                        print(exp['message'])
                return exp

            def get_reward(self):
                reward = super().get_reward()
                attr = getattr(self.mdp, PATH_FUNC)
                if self is attr.controller:
                    _attr = getattr(self, PATH_FUNC)
                    _attr.exp['reward'] = reward
                    attr.socket.send(msgpack.dumps(('append', (_attr.exp, _attr.state), {}), default=lamarckian.util.serialize.encode))
                    attr.socket.recv()
                return reward

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            port = port_for.select_random(glom.glom(kwargs['config'], 'rpc.ports', default=None))
            context = zmq.Context()
            socket = context.socket(zmq.REQ)
            socket.bind(f"tcp://*:{port}")
            encoding = self.describe()
            proc = multiprocessing.get_context('spawn').Process(target=functools.partial(run, encoding, host=ray.util.get_node_ip_address(), port=port, **kwargs))
            proc.start()
            setattr(self, PATH_FUNC, types.SimpleNamespace(
                proc=proc,
                socket=socket,
                encoding=encoding,
            ))

        def close(self):
            super().close()
            attr = getattr(self, PATH_FUNC)
            attr.socket.send(msgpack.dumps((None, (), {}), default=lamarckian.util.serialize.encode))
            attr.socket.recv()
            attr.proc.join()

        def reset(self, *args, **kwargs):
            battle = super().reset(*args, **kwargs)
            getattr(self, PATH_FUNC).controller = battle.controllers[0]
            return battle
    return MDP
