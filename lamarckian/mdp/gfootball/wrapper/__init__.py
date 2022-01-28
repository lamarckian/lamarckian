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

import gym


class PeriodicDumpWriter(gym.Wrapper):
    """A wrapper that only dumps traces/videos periodically."""

    def __init__(self, env, dump_frequency):
        gym.Wrapper.__init__(self, env)
        self._dump_frequency = dump_frequency
        self._original_dump_config = {
            'write_video': env._config['write_video'],
            'dump_full_episodes': env._config['dump_full_episodes'],
            'dump_scores': env._config['dump_scores'],
        }
        self._current_episode_number = 0

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        if (self._dump_frequency > 0 and
                (self._current_episode_number % self._dump_frequency == 0)):
            self.env._config.update(self._original_dump_config)
            self.env.render()
        else:
            self.env._config.update({'write_video': False,
                                     'dump_full_episodes': False,
                                     'dump_scores': False})
            self.env.disable_render()
        self._current_episode_number += 1
        return self.env.reset()
