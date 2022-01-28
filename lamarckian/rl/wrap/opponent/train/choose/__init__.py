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

import inspect
import itertools
import types
import traceback

import numpy as np
import glom
import matplotlib.pyplot as plt

import lamarckian


def random(rl):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    class RL(rl):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            setattr(self, PATH_FUNC, types.SimpleNamespace(rs=np.random.RandomState(glom.glom(kwargs['config'], 'seed', default=None))))

        def set_opponents_train(self, opponents, *args, **kwargs):
            attr = getattr(self, PATH_FUNC)
            attr.__dict__.pop('probs', None)
            return super().set_opponents_train(opponents, *args, **kwargs)

        def set_choose_opponent_train_prob(self, probs):
            assert len(self.get_opponents_train()) == len(probs), (len(self.get_opponents_train()), len(probs))
            attr = getattr(self, PATH_FUNC)
            attr.probs = probs

        def _choose_opponent_train(self):
            attr = getattr(self, PATH_FUNC)
            opponents = list(self._get_opponents_train().values())
            probs = getattr(attr, 'probs', None)
            try:
                return attr.rs.choice(opponents, p=probs)
            except ValueError:
                traceback.print_exc()
                return attr.rs.choice(opponents)
    return RL


def cycle(rl):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    class RL(rl):
        def set_opponents_train(self, *args, **kwargs):
            try:
                return super().set_opponents_train(*args, **kwargs)
            finally:
                setattr(self, PATH_FUNC, itertools.cycle(list(self._get_opponents_train().values())))

        def _choose_opponent_train(self):
            return next(getattr(self, PATH_FUNC))
    return RL


class RecordPFSP(object):
    def __init__(self, tag, cost, color, **kwargs):
        self.tag = tag
        self.cost = cost
        self.color = color
        self.kwargs = kwargs

    def __call__(self, recorder):
        for key, value in self.kwargs.items():
            fig = plt.figure()
            ax = fig.gca()
            x = np.arange(len(value))
            ax.bar(x, value, color=self.color)
            fig.tight_layout()
            fig.canvas.draw()
            image = lamarckian.util.mpl.to_image(fig)
            plt.close(fig)
            recorder.writer.add_image(f'{self.tag}/{key}', np.transpose(image, [2, 0, 1]), self.cost)


def pfsp(rl):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    def fetch(self):
        opponents = self._get_opponents_train().values()
        color = ['r' if any(isinstance(blob, str) for blob in opponent['blobs'].values()) else 'b' for opponent in opponents]
        total = np.array([len(opponent['win']) for opponent in opponents])
        win = np.array([float(opponent['win']) for opponent in opponents])
        probs = self.get_pfsp_probs(opponents)
        return dict(color=color, total=total, win=win, probs=probs)

    class RL(rl):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            setattr(self, PATH_FUNC, types.SimpleNamespace(
                scale=eval('lambda x: ' + glom.glom(kwargs['config'], f'rl.opponent.train.{NAME_FUNC}.scale')),
                sample=glom.glom(kwargs['config'], f"rl.opponent.train.{NAME_FUNC}.sample", default=9),
                rs=np.random.RandomState(glom.glom(kwargs['config'], 'seed', default=None)),
            ))
            if hasattr(self, 'recorder'):
                self.recorder.register(
                    lamarckian.util.counter.Time(**glom.glom(kwargs['config'], 'record.plot')),
                    lambda *args, **kwargs: RecordPFSP(f'opponents/{NAME_FUNC}', self.cost, **fetch(self)),
                )

        def _choose_opponent_train(self):
            attr = getattr(self, PATH_FUNC)
            opponents = list(self._get_opponents_train().values())
            probs = self.get_pfsp_probs(opponents)
            return attr.rs.choice(opponents, p=probs)

        def get_pfsp_probs(self, opponents):
            attr = getattr(self, PATH_FUNC)
            win = np.array([float(opponent['win']) for opponent in opponents])
            probs = win.max() - win
            probs = attr.scale(probs)
            mask = np.array([len(opponent['win']) for opponent in opponents]) < attr.sample
            if mask.any():
                probs[mask] = probs.max()
            return lamarckian.util.to_probs(probs)
    return RL


def openai_five(rl):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    import random

    class RL(rl):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, PATH_FUNC)
            setattr(self, PATH_FUNC, types.SimpleNamespace(
                prob=glom.glom(kwargs['config'], f'rl.opponent.train.{NAME_FUNC}.prob'),
                random=random.Random(glom.glom(kwargs['config'], 'seed', default=None)),
            ))

        def _choose_opponent_train(self):
            attr = getattr(self, PATH_FUNC)
            items = list(self._get_opponents_train().values())
            if attr.random.random() < attr.prob:
                return items[-1]
            else:
                items_former = items[:-1]
                if items_former:
                    return attr.random.choice(items_former)
                else:
                    return items[-1]
    return RL
