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

import itertools
import operator
import math

try:
    import nnpy
    import pynng
except ImportError:
    pass
import toolz.itertoolz
import ray.services


class Node(object):
    def __init__(self, host, actors):
        self.host = host
        self.actors = actors

    def __iter__(self):
        return iter(self.actors)

    def __getitem__(self, item):
        return self.actors[item]

    def __len__(self):
        return len(self.actors)

    def __repr__(self):
        return f"{self.host} with {len(self.actors)} actors"


class Topology(object):
    def __init__(self, actors, degree=None, local=True):
        self.host = ray.services.get_node_ip_address()
        if local:
            self.local = {actor for actor, host in actors.items() if host == self.host}
        else:
            self.local = {}
        remote = {actor: host for actor, host in actors.items() if actor not in self.local}
        assert len(self.local) + len(remote) == len(actors), (len(self.local), len(remote), len(actors))
        self.nodes = {Node(host, list(map(operator.itemgetter(0), grouped))) for host, grouped in toolz.groupby(operator.itemgetter(1), remote.items()).items()}
        assert len(set(itertools.chain(*self.nodes))) == len(remote)
        if degree is None:
            degree = math.ceil(math.sqrt(len(self.nodes)))
        self.relays = {nodes[0]: nodes[1:] for nodes in toolz.itertoolz.partition_all(degree, self.nodes)}
        assert len(set(itertools.chain(*([relay] + list(nodes) for relay, nodes in self.relays.items())))) == len(self.nodes)
