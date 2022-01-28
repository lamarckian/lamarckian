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

import time

import numpy as np
import glom
import psutil

import lamarckian


class Profiler(object):
    def __init__(self, cost, size=1, tag='profile', **kwargs):
        self.cost = cost
        self.size = size
        self.tag = tag
        self.kwargs = kwargs
        self.time = time.time()
        self.net_io = psutil.net_io_counters()

    def get(self, cost):
        t = time.time()
        elapsed = t - self.time + np.finfo(np.float).eps
        speed = (cost - self.cost) / elapsed
        net_io = psutil.net_io_counters()
        try:
            data = {
                **{
                    f"{self.tag}/cost": speed,
                },
                **{f"{self.tag}/net_io/{key}": (getattr(net_io, key) - getattr(self.net_io, key)) / elapsed for key in glom.glom(self.kwargs['config'], 'profile.net_io', default=[])},
            }
            if self.size > 1:
                data[f"{self.tag}/cost/avg"] = speed / self.size
            try:
                with open('/proc/sys/net/netfilter/nf_conntrack_count') as f_count, open('/proc/sys/net/netfilter/nf_conntrack_max') as f_max:
                    data[f"{self.tag}/conntrack"] = int(f_count.read()) / int(f_max.read())
            except FileNotFoundError:
                pass
            return data
        finally:
            self.cost = cost
            self.time = t
            self.net_io = net_io

    def __call__(self, cost):
        return lamarckian.util.record.Scalar(cost, **self.get(cost))
