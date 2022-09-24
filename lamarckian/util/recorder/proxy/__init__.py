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
import threading
import contextlib

import zmq

from lamarckian.util.recorder import Process


class Client(Process):
    class Worker(Process.Worker):
        def __init__(self, **kwargs):
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect(kwargs['recorder'])

        def close(self):
            self.socket.send(b'')
            self.socket.recv()
            self.socket.close()
            self.context.term()

        def __call__(self):
            record = self.queue.get()
            self.socket.send(pickle.dumps(record))
            self.socket.recv()
            return record


class Server(Process):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        def run():
            context = zmq.Context()
            with contextlib.closing(context.socket(zmq.REP)) as socket:

                socket.bind(f"tcp://*:{kwargs['port']}")
                while True:
                    data = socket.recv()
                    socket.send(b'')
                    if not data:
                        break
                    record = pickle.loads(data)
                    self.put(record)
            context.term()
        worker = threading.Thread(target=run)
        worker.start()
