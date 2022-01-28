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
import collections
import numbers
import contextlib
import operator
import traceback
import logging

import numpy as np
import torch
import glom
import tqdm
import humanfriendly
import imageio
import imagecodecs
import matplotlib.pyplot as plt

import lamarckian


NL = '\n'
NLNL = '\n\n'


class Graph(object):
    def __init__(self, tag, cost):
        self.tag = tag
        self.cost = cost

    def __call__(self, recorder):
        if hasattr(recorder, 'evaluator'):
            evaluator = recorder.evaluator
            encoding = evaluator.describe()['blob']
            cls = lamarckian.evaluator.parse(*encoding['agent']['eval'], **evaluator.kwargs)
            agent = cls(evaluator.model, **evaluator.kwargs)
            model = encoding['models'][evaluator.me]
            inputs = agent.get_inputs(dict(inputs=tuple(np.zeros(input['shape'], dtype=np.float32) for input in model['inputs'])))

            def wrap(func):
                def _(*args, **kwargs):
                    outputs = func(*args, **kwargs)
                    return *outputs.values(),
                return _
            with contextlib.closing(lamarckian.util.restoring.attr(evaluator.model, forward=wrap(evaluator.model.forward))):
                recorder.writer.add_graph(evaluator.model, inputs)
            recorder.writer.add_text(self.tag, humanfriendly.format_size(sum(value.numpy().nbytes for value in evaluator.model.state_dict().values())), self.cost)


class Model(object):
    HEADER = '| layer | shape | bytes | size | density |\n| :-- | :-- | :-- | :-- | :-- |\n'

    def __init__(self, tag, cost, blob):
        self.tag = tag
        self.cost = cost
        self.blob = blob

    def __call__(self, recorder):
        if hasattr(recorder, 'evaluator'):
            evaluator = recorder.evaluator
            model = evaluator.model
            state_dict = lamarckian.model.from_blob(self.blob, model.state_dict())
            layers = [dict(key=key, value=value, density=value.abs().mean().item()) for key, value in state_dict.items()]
            layers = sorted(layers, key=operator.itemgetter('density'), reverse=True)
            recorder.writer.add_text(f"{self.tag}/size", self.HEADER + NL.join([f"| {layer['key']} | {'x'.join(map(str, layer['value'].shape))} | {layer['value'].numpy().nbytes} | {humanfriendly.format_size(layer['value'].numpy().nbytes)} | {layer['density']} |" for layer in layers]), self.cost)


class Freq(object):
    def __init__(self, cost, freq, tag='freq', names=None, **kwargs):
        self.cost = cost
        self.freq = freq
        self.tag = tag
        self.names = list(map(str, range(len(freq)))) if names is None else names
        self.kwargs = kwargs

    def __call__(self, recorder):
        fig = plt.figure(**self.kwargs)
        ax = fig.gca()
        ax.cla()
        x = np.arange(len(self.freq))
        self.plot(ax, x)
        ax.set_xticks(x)
        ax.set_xticklabels(self.names)
        fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
        fig.tight_layout()
        fig.canvas.draw()
        image = lamarckian.util.mpl.to_image(fig)
        recorder.writer.add_image(self.tag, np.transpose(image, [2, 0, 1]), self.cost)

    def plot(self, ax, x):
        assert len(self.freq.shape) == 2, self.freq.shape
        fail, success = self.freq.T
        total = success + fail
        rects = ax.bar(x, total, label='total')
        ax.bar(x, success, label='success')
        for _success, _total, rect in zip(success, total, rects):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), '%.1f/%.1f' % (_success, _total), ha='center', va='bottom')


class Rollout(object):
    HEADER_DISCRETE = '| action | success | total |\n| :-- | :-- | :-- |\n'

    def __init__(self, cost, blob, tag='rollout'):
        self.cost = cost
        self.blob = blob
        self.tag = tag

    def dump(self, recorder, prefix, trajectory, exp, result, **kwargs):
        if glom.glom(kwargs['config'], 'record.evaluate.save', default=True):
            path = os.path.dirname(prefix) + '.mdp.pth'
            if not os.path.exists(path):
                torch.save(self.state, path)
            torch.save(dict(trajectory=trajectory, exp=exp, result=result), prefix + '.pth')
        if 'image' in trajectory[0]['state']:
            fps = glom.glom(kwargs['config'], 'mdp.fps')
            path = prefix + '.mp4'
            with imageio.get_writer(path, fps=fps) as writer:
                for exp in tqdm.tqdm(trajectory, desc=f'save {path}'):
                    image = exp['state']['image']
                    if not isinstance(image, np.ndarray):
                        image = imagecodecs.jpeg_decode(image)
                    writer.append_data(image)
            with open(prefix + '.txt', 'w') as f:
                f.write(f"{result}")
        if glom.glom(kwargs['config'], 'record.evaluate.reward', default=True):
            encoding = self.state['encoding']['blob']
            fig = plt.figure()
            ax = fig.gca()
            discount = glom.glom(kwargs['config'], 'rl.discount')
            for i, name in enumerate(tqdm.tqdm(encoding['reward'], desc=f"dump reward")):
                reward = np.array([exp['reward'][i] for exp in trajectory])
                ax.cla()
                ax.plot(*zip(*enumerate(reward)))
                fig.tight_layout()
                fig.canvas.draw()
                image = lamarckian.util.mpl.to_image(fig)
                recorder.writer.add_image(f"{self.tag}/reward/{name}", np.transpose(image, [2, 0, 1]), self.cost)
                # credit
                _discount = glom.glom(kwargs['config'], f"rl.discount_{name}", default=discount)
                credit = lamarckian.rl.cumulate(reward, _discount)
                ax.cla()
                ax.plot(*zip(*enumerate(reward)))
                ax.plot(*zip(*enumerate(credit)))
                ax.set_title(str(_discount))
                fig.tight_layout()
                fig.canvas.draw()
                image = lamarckian.util.mpl.to_image(fig)
                recorder.writer.add_image(f"{self.tag}/credit/{name}", np.transpose(image, [2, 0, 1]), self.cost)
            plt.close(fig)
        if glom.glom(kwargs['config'], 'record.evaluate.discrete', default=True):
            encoding = self.state['encoding']['blob']
            for i, names in enumerate(tqdm.tqdm(encoding['models'][encoding['me']].get('discrete', []), desc=f"dump discrete")):
                total = [exp['discrete'][i].item() for exp in trajectory]
                success = [exp['discrete'][i].item() for exp in trajectory if not exp.get('error', '')]
                total, success = np.bincount(total, minlength=len(names)), np.bincount(success, minlength=len(names))
                recorder.writer.add_text(f"{self.tag}/discrete{i}", self.HEADER_DISCRETE + NL.join([f"| {name} | {success} | {total} |" for name, success, total in zip(names, success, total)]), self.cost)

    def __call__(self, recorder):
        if hasattr(recorder, 'evaluator'):
            evaluator = recorder.evaluator
            kwargs = evaluator.kwargs
            evaluator.cost = self.cost
            evaluator.set_blob(self.blob)
            root = os.path.join(os.path.dirname(os.path.dirname(recorder.writer.log_dir)), 'rollout', str(self.cost))
            os.makedirs(root, exist_ok=True)
            results = []
            sample = glom.glom(kwargs['config'], 'sample.eval')
            self.state = evaluator.__getstate__()
            for seed, cost, trajectory, exp, result in tqdm.tqdm(evaluator.iterate_trajectory(sample, message=True), total=sample, desc='rollout'):
                try:
                    self.dump(recorder, os.path.join(root, str(seed)), trajectory, exp, result, **kwargs)
                except:
                    traceback.print_exc()
                results.append(result)
            result = evaluator.reduce(results)
            _result = {key: value for key, value in result.items() if not key.startswith('_') and isinstance(value, numbers.Number)}
            logging.info(_result)
            for key, value in _result.items():
                recorder.writer.add_scalar(f"{self.tag}/{key}", value, self.cost)
            torch.save(dict(decision=dict(blob=self.blob)), root + '.pth')
            with open(f"{root}.result{len(results)}", 'w') as f:
                f.write(f"{_result}")

    @staticmethod
    def counter(**kwargs):
        return lamarckian.util.counter.Time(**{key: value for key, value in glom.glom(kwargs['config'], 'record.evaluate', default=dict(interval=np.nan)).items() if key in {'interval', 'first'}})

    @staticmethod
    def new(**kwargs):
        try:
            return lamarckian.evaluator.parse(*glom.glom(kwargs['config'], 'record.rollout.create'), **kwargs)
        except KeyError:
            return Rollout


class Results(collections.deque):
    def __init__(self, tag='result/train', **kwargs):
        self.tag = tag
        super().__init__(maxlen=glom.glom(kwargs['config'], 'sample.train'))

    def __call__(self):
        if self:
            return {f"{self.tag}/{key}": np.mean([result[key] for result in self]) for key, value in self[0].items() if not key.startswith('_') and isinstance(value, numbers.Number)}
        else:
            return {}
