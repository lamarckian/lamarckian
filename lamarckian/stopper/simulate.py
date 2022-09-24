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
import argparse
import logging
import logging.config
import contextlib
import collections
import types
import tempfile
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import humanfriendly
import glom
import tqdm
import oyaml

import lamarckian


class Writer(object):
    def __init__(self, data):
        self.data = data

    def add_scalar(self, key, value, x):
        if key not in self.data:
            self.data[key] = collections.OrderedDict()
        self.data[key][x] = value


class Recorder(object):
    def __init__(self):
        self.data = {}
        self.writer = Writer(self.data)

    def put(self, record):
        record(self)


class Evaluator(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.recorder = Recorder()

    def set(self, *args, **kwargs):
        pass

    def get(self):
        return {}

    def evaluate(self):
        return self.result

    def __getstate__(self):
        return {}


def main():
    args = make_args()
    config = {}
    for path in sum(args.config, []):
        config = lamarckian.util.config.read(path, config)
    for cmd in sum(args.modify, []):
        lamarckian.util.config.modify(config, cmd)
    logging.config.dictConfig(config['logging'])
    root = os.path.expanduser(os.path.expandvars(args.root))
    with tempfile.TemporaryDirectory() as root_log:
        logging.info(root_log)
        for dirpath, dirnames, filenames in os.walk(root):
            for filename in filenames:
                if filename.endswith('.csv'):
                    path = os.path.join(dirpath, filename)
                    logging.info(path)
                    df = pd.read_csv(path)
                    _df = df[args.begin:args.end]
                    if _df.empty:
                        logging.warning(f'df[{args.begin}:{args.end}] is empty (len(df) == {len(df)})')
                        continue
                    costs = _df['Step'].to_numpy()
                    values = _df['Value'].to_numpy()
                    times = _df['Wall time'].to_numpy()
                    evaluator = Evaluator(config=config, root=root)
                    Stopper = lamarckian.evaluator.parse(*glom.glom(config, 'ec.train.stopper'), config=config)
                    terminated = None
                    evaluator.cost = int(os.path.splitext(filename)[0])
                    with contextlib.closing(Stopper(evaluator, config=config)) as stopper:
                        for i, (cost, value) in enumerate(zip(costs, tqdm.tqdm(values))):
                            result = {args.result: value}
                            outcome = types.SimpleNamespace(cost=cost - evaluator.cost, result=result)
                            evaluator.result = result
                            try:
                                if stopper(outcome):
                                    terminated = i
                                    break
                            finally:
                                evaluator.cost = cost
                    fig = plt.figure()
                    ax = fig.gca()
                    if args.original:
                        ax.plot(costs, values)
                        if len(values) <= args.points:
                            improve = np.concatenate([[0], values[1:] - values[:-1]])
                            ax.scatter(costs, values, c=np.sign(improve), cmap='seismic', picker=args.picker)

                            def on_pick(event):
                                for i in event.ind:
                                    print(f'point{i}: cost={costs[i]}, value={values[i]}, improve={improve[i]}')
                            fig.canvas.mpl_connect('pick_event', on_pick)
                    if terminated is not None:
                        x = costs[terminated]
                        ymin, ymax = values.min(), values.max()
                        if ymin < ymax:
                            ax.plot([x, x], [ymin, ymax], '--')
                        else:
                            ax.plot(x, values[terminated], 'x')
                        ax.set_xlabel(f'{terminated}: {humanfriendly.format_timespan(times[terminated] - times[args.begin])}')
                    for key, data in evaluator.recorder.data.items():
                        ax.plot(data.keys(), data.values(), label=key)
                    ax.set_title(f'{costs[-1] - costs[0]}: {humanfriendly.format_timespan(times[-1] - times[0])}')
                    fig.tight_layout()
                    fig.canvas.manager.set_window_title(path)
                    if args.ext:
                        text = oyaml.dump(config, allow_unicode=True)
                        digest = hashlib.md5(text.encode() + costs.tostring() + values.tostring()).hexdigest()
                        prefix = f'{os.path.splitext(path)[0]}.{digest}'
                        logging.info(prefix)
                        with open(prefix + '.yml', 'w') as f:
                            f.write(text)
                        plt.savefig(prefix + args.ext)
                    if args.show:
                        plt.show()
                    plt.close(fig)


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('-c', '--config', nargs='+', action='append', default=[[os.path.dirname(os.path.abspath(lamarckian.__file__)) + '.yml']])
    parser.add_argument('-m', '--modify', nargs='+', action='append', default=[])
    parser.add_argument('-r', '--result', default='fitness')
    parser.add_argument('-b', '--begin', type=int, default=0)
    parser.add_argument('-e', '--end', type=int, default=np.iinfo(np.int).max)
    parser.add_argument('-o', '--original', action='store_false')
    parser.add_argument('-s', '--show', action='store_true')
    parser.add_argument('--ext', default='.svg')
    parser.add_argument('--points', type=int, default=50)
    parser.add_argument('--picker', type=int, default=3)
    return parser.parse_args()


if __name__ == '__main__':
    main()
