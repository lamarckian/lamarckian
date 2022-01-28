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
import re
import functools
import contextlib
import logging

import yaml
import metayaml
import glom
import deepmerge

import lamarckian

PROG = re.compile(r'\$\{([a-zA-Z0-9_\.]+)\}')
DEFAULTS = dict(
    os=os,
    ROOT=os.sep.join(os.path.abspath(lamarckian.__file__).split(os.sep)[:-2]),
)


@contextlib.contextmanager
def push_cwd(root):
    _root = os.getcwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(_root)


@contextlib.contextmanager
def push_attr(config, **kwargs):
    old = {key: config[key] for key in kwargs if key in config}
    for key, value in kwargs.items():
        config[key] = value
    try:
        yield
    finally:
        for key in kwargs:
            if key in old:
                config[key] = old[key]
            else:
                del config[key]


class MetaYaml(metayaml.MetaYaml):
    def load(self, path, data):
        with push_cwd(os.path.dirname(path)), push_attr(data, PATH=path):
            assert hasattr(self, 'processed_files')
            self.processed_files = set()
            return super().load(path, data)


def read(path, config={}, **kwargs):
    kwargs = {**DEFAULTS, **kwargs}
    for key, value in kwargs.items():
        config[key] = value
    try:
        config = MetaYaml(os.path.expanduser(os.path.expandvars(path)), defaults=config).data
    except metayaml.metayaml.MetaYamlException:
        logging.fatal(path)
        raise
    for key in kwargs:
        del config[key]
    return config


def modify(config, cmd):
    try:
        select, value = cmd.split('=', 1)
    except ValueError:
        logging.fatal(cmd)
        raise
    if value:
        value = PROG.sub(lambda m: str(functools.reduce(lambda x, key: x[key], [config] + m.group(1).split('.'))), value)
        value = yaml.safe_load(f'x: {value}')['x']
        glom.assign(config, select, value, missing=dict)
    else:
        glom.delete(config, select, ignore_missing=True)
    return config


def make_test(config, **kwargs):
    for path in glom.glom(config, 'test_read', default=[]):
        config = read(path, config, **kwargs)
    for cmd in glom.glom(config, 'test_modify', default=[]):
        config = modify(config, cmd)
    return deepmerge.always_merger.merge(config, glom.glom(config, 'test', default={}))
