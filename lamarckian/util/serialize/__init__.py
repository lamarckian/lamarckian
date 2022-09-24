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

import pickle
import lzma

import torch
import msgpack
import msgpack_numpy
import pyarrow
import lz4framed


def encode(obj, *args, **kwargs):
    if torch.is_tensor(obj):
        data = msgpack_numpy.encode(obj.cpu().numpy(), *args, **kwargs)
        data['pth'] = True
        return data
    else:
        return msgpack_numpy.encode(obj, *args, **kwargs)


def decode(data, *args, **kwargs):
    if data.pop('pth', False):
        return torch.from_numpy(msgpack_numpy.decode(data, *args, **kwargs))
    else:
        return msgpack_numpy.decode(data, *args, **kwargs)


def msgpack_dumps(value):
    try:
        return msgpack.dumps(value, default=encode)
    except:
        return pickle.dumps(value)


def msgpack_loads(value):
    try:
        return msgpack.loads(value, object_hook=decode, strict_map_key=False)
    except:
        return pickle.loads(value)


def pyarrow_dumps(value):
    try:
        return pyarrow.serialize(value).to_buffer()
    except:
        return pickle.dumps(value)


def pyarrow_loads(value):
    try:
        return pyarrow.deserialize(value)
    except:
        return pickle.loads(value)


SERIALIZE = dict(
    pickle=pickle.dumps,
    pickle_lz4=lambda value: lz4framed.compress(pickle.dumps(value)),
    msgpack=msgpack_dumps,
    msgpack_lz4=lambda value: lz4framed.compress(msgpack_dumps(value)),
    msgpack_lzma=lambda value: lzma.compress(msgpack_dumps(value)),
    pyarrow=pyarrow_dumps,
    pyarrow_lz4=lambda value: lz4framed.compress(pyarrow.serialize(value).to_buffer()),
)

DESERIALIZE = dict(
    pickle=pickle.loads,
    pickle_lz4=lambda value: pickle.loads(lz4framed.decompress(value)),
    msgpack=msgpack_loads,
    msgpack_lz4=lambda value: msgpack_loads(lz4framed.decompress(value)),
    msgpack_lzma=lambda value: msgpack_loads(lzma.decompress(value)),
    pyarrow=pyarrow_loads,
    pyarrow_lz4=lambda value: pyarrow_loads(lz4framed.decompress(value)),
)


class Serializer(object):
    def __init__(self, key):
        self.key = key
        self.serialize = SERIALIZE[key]
        self.deserialize = DESERIALIZE[key]

    def __repr__(self):
        return self.key
