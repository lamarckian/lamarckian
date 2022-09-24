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

import enum
import itertools

import tqdm
import ray.exceptions

LOCK_PORT = '/tmp/lamarckian.port'
IPC_PREFIX = f"/tmp/lamarckian.rpc."
ID_LENGTH = 9
SIZE_BYTES = 4
Fetch = enum.Enum('Fetch', zip('call any all'.split(), itertools.count()))


class Node(list):
    def __init__(self, host, actors):
        self.host = host
        super().__init__(actors)

    def __hash__(self):
        return hash(self.host)

    def __repr__(self):
        return f"{self.host} with {len(self)} actors"


def detect_died_actors(actors, check):
    ips = []
    for actor in tqdm.tqdm(actors, desc='detect died actors'):
        try:
            ray.get(check(actor))
        except ray.exceptions.RayActorError:
            ips.append(ray.actors(actor._actor_id.hex())['Address']['IPAddress'])
    return ips
