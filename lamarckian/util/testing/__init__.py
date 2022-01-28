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

import os
import io
import zipfile
import logging

import numpy as np
import torch


class Scope(object):
    def __init__(self, prefix, ext='.zip'):
        path = prefix + ext
        if os.path.exists(path):
            self.archive = zipfile.ZipFile(path, 'r')
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.archive = zipfile.ZipFile(path, 'w')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.archive.close()
        if exc_type is not None and self.archive.mode == 'w':
            os.remove(self.archive.filename)

    def assert_array_almost_equal(self, prefix, value, ext='.pth', **kwargs):
        path = '/' + prefix + ext
        if self.archive.mode == 'r':
            buffer = io.BytesIO(self.archive.read(path))
            np.testing.assert_array_almost_equal(torch.load(buffer), value, **kwargs)
        else:
            buffer = io.BytesIO()
            torch.save(value, buffer)
            self.archive.writestr(path, buffer.getvalue())
            logging.warning(f'save {path}')
