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

import torch
import glom


def reset(mdp):
    class MDP(mdp):
        def reset(self, *args, **kwargs):
            try:
                torch.manual_seed(glom.glom(self.kwargs['config'], 'seed'))
            except KeyError:
                pass
            return super().reset(*args, **kwargs)
    return MDP


def cast(mdp):
    class MDP(mdp):
        class Controller(mdp.Controller):
            async def __call__(self, *args, **kwargs):
                try:
                    torch.manual_seed(glom.glom(self.mdp.kwargs['config'], 'seed'))
                except KeyError:
                    pass
                return await super().__call__(*args, **kwargs)
    return MDP
