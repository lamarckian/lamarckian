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

from . import yegor_rule


class Builtin(object):
    def __init__(self, controller):
        self.controller = controller

    async def __call__(self):
        exp = await self.controller('builtin_ai')
        return exp['done']


class Yegor(object):
    def __init__(self, controller):
        self.controller = controller
        self.env = controller.mdp.env

    async def __call__(self):
        observation = self.env.unwrapped.observation()[self.controller.me]
        action, = yegor_rule.agent(observation)
        exp = await self.controller(action)
        return exp['done']
