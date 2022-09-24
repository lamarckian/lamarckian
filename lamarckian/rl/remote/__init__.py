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

import contextlib
import threading

import lamarckian


class Broadcaster(object):
    def __init__(self, actors, rpc=lamarckian.util.rpc.All, **kwargs):
        self.actors = actors
        self.rpc = rpc
        self.kwargs = kwargs
        self.event = threading.Event()
        self.running = True
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def close(self):
        self.running = False
        thread = threading.Thread(target=self.thread.join)
        thread.start()
        while thread.is_alive():
            self.event.set()
        thread.join()

    def __call__(self):
        self.event.set()

    def broadcast(self, rpc):
        raise NotImplementedError()

    def run(self):
        with contextlib.closing(self.rpc(self.actors, **self.kwargs)) as rpc:
            while self.running:
                self.event.wait()
                self.event.clear()
                self.broadcast(rpc)
