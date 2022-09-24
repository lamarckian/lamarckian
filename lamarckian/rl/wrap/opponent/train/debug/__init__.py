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

import inspect
import hashlib
import pickle
import logging


def digest(rl):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    class RL(rl):
        def _set_opponent_train(self, item, *args, **kwargs):
            if getattr(self, PATH_FUNC):
                digest = {kind: hashlib.md5(pickle.dumps(blob)).hexdigest() for kind, blob in item.blobs.items()}
                logging.warning(f'set opponent_train={digest}')
            return super()._set_opponent_train(item, *args, **kwargs)

        def _get_opponent_train(self):
            item = super()._get_opponent_train()
            if getattr(self, PATH_FUNC):
                digest = {kind: hashlib.md5(pickle.dumps(blob)).hexdigest() for kind, blob in item.blobs.items()}
                logging.warning(f'get opponent_train (agent)={digest}')
            return item
    return RL
