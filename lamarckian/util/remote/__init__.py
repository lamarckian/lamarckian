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

import collections.abc

import tqdm
import ray.exceptions


def submit(actors, tasks, ray=ray):
    assert isinstance(tasks, collections.abc.Iterator)
    running = [creator(actor) for actor, creator in zip(actors, tasks)]
    done = []
    for task in tasks:
        (ready,), _ = ray.wait(running)
        index = running.index(ready)
        actor = actors[index]
        running[index] = task(actor)
        done.append(ready)
    return done + running


def progress(tasks, desc=None, ray=ray):
    with tqdm.tqdm(total=len(tasks), desc=desc) as pbar:
        while tasks:
            _, tasks = ray.wait(tasks)
            pbar.update(1)
    return tasks


class AsyncCall(object):
    def __init__(self, actors, call, ray=ray):
        self.actors = actors
        self.call = call
        self.ray = ray
        self.task = [self.call(actor, None) for actor in self.actors]

    def close(self):
        self.ray.get(self.task)

    def get_idle(self):
        if self.task:
            (ready,), _ = self.ray.wait(self.task)
            return self.actors[self.task.index(ready)]
        else:
            return self.actors[0]

    def __call__(self):
        (ready,), _ = self.ray.wait(self.task)
        index = self.task.index(ready)
        actor = self.actors[index]
        result = self.ray.get(ready)
        self.task[index] = self.call(actor, result)
        return result


class RayActorError(ray.exceptions.RayActorError):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg
