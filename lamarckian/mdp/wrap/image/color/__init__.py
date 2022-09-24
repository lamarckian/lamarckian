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


def grey_hwc(index):
    import cv2

    def decorate(mdp):
        class MDP(mdp):
            class Controller(mdp.Controller):
                def get_state(self):
                    state = super().get_state()
                    state['inputs'][index] = cv2.cvtColor(state['inputs'][index], cv2.COLOR_BGR2GRAY)
                    return state

            def describe_blob(self):
                encoding = super().describe_blob()
                for model in encoding['models']:
                    model['inputs'][index]['shape'] = model['inputs'][index]['shape'][:-1]
                return encoding
        return MDP
    return decorate


def scale(index, min, max):
    assert min < max, (min, max)
    range = max - min

    def decorate(mdp):
        class MDP(mdp):
            class Controller(mdp.Controller):
                def get_state(self):
                    state = super().get_state()
                    state['inputs'][index] = (state['inputs'][index] - min) / range
                    return state
        return MDP
    return decorate
