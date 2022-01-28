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

import lamarckian


def all(name='rpc_all'):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    NL = '\n'
    NLNL = '\n\n'
    TABLE_HEADER = '| node | actors |\n| :-- | :-- |\n'

    def decorate(rl):
        class RL(rl):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                if 'ray' not in kwargs:
                    relay = getattr(self, name).topology.relay
                    self.recorder.put(lamarckian.util.record.Text(self.cost, **{f'{name}/{NAME_FUNC}': NLNL.join(f'# {node.host} ({len(node.actors)} actors, {len(nodes)} nodes)\n{TABLE_HEADER + NL.join([f"| {node.host} | {len(node)} |" for node in nodes])}' for node, nodes in relay.items())}))
        return RL
    return decorate
