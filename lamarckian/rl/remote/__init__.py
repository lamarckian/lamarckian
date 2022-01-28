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

import threading


class Runner(object):
    def __init__(self, func):
        super().__init__()
        self.func = func
        self.running = True
        self.event = threading.Event()
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

    def run(self):
        while self.running:
            self.event.wait()
            self.func()
