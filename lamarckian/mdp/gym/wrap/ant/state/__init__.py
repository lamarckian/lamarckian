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


def scale(mdp):
    class MDP(mdp):
        class Controller(mdp.Controller):
            def get_state(self):
                state = super().get_state()
                input, = state['inputs']
                # 对input标准化
                state['inputs'] = [input]
                return state
    return MDP
