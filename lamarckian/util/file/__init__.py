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

import os
import shutil
import hashlib
import operator


def tidy(root, keep=5, ext='.pth'):
    prefixes = {os.path.splitext(name)[0] for name in os.listdir(root) if name.endswith(ext)}
    prefixes = [int(prefix) for prefix in prefixes if prefix.isdigit()]
    if len(prefixes) > keep:
        prefixes = sorted(prefixes)
        remove = prefixes[:-keep]
        for prefix in map(str, remove):
            for name in os.listdir(root):
                if name.startswith(prefix):
                    path = os.path.join(root, name)
                    if os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        os.remove(path)


def load(root, ext='.pth'):
    for n in sorted([int(n) for n, e in map(os.path.splitext, os.listdir(root)) if n.isdigit() and e == ext], reverse=True):
        yield os.path.join(root, f"{n}{ext}")


def group_filename(comp, concat='-'):
    if len(comp) == 1:
        return comp[0]
    prefix = os.path.commonprefix(comp)
    if prefix:
        comp = [s[len(prefix):] for s in comp]
        return f'{prefix}[{concat.join([comp[0], group_filename(comp[1:], concat)])}]'
    else:
        return concat.join([comp[0], group_filename(comp[1:], concat)])


def short_filename(name, max=255, join='|'):
    if len(name) > max:
        digest = hashlib.md5(name.encode()).hexdigest()
        return name[:max - len(join) - len(digest)] + join + digest
    else:
        return name
