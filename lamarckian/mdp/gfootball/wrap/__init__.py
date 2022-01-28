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

import traceback

from . import state


def fault_tolerant(mdp):
    class MDP(mdp):
        class Controller(mdp.Controller):
            async def __call__(self, *args, **kwargs):
                try:
                    return await super().__call__(*args, **kwargs)
                except KeyboardInterrupt:
                    raise
                except GeneratorExit:
                    pass
                except:
                    traceback.print_exc()
                    try:
                        self.mdp.env.close()
                    except:
                        traceback.print_exc()
                    self.mdp.env = self.mdp.create_env()
                    self.mdp.env.reset()
                    return await super().__call__(*args, **kwargs)
    return MDP


def env_repr(mdp):
    class MDP(mdp):
        class Controller(mdp.Controller):
            def __repr__(self):
                return repr(self.mdp.env)
    return MDP
