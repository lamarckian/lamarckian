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
import sys
import copy
import multiprocessing
import threading
import queue
import subprocess
import signal
import datetime
import time
import tarfile
import platform
import traceback
import logging
import asyncio

import torch.utils.tensorboard
import glom
import tqdm
import humanfriendly
import oyaml
import difflib
import setproctitle
import pathspec

import lamarckian


def backup(root, dst):
    relpaths = [os.path.relpath(os.path.join(dirpath, filename), root) for dirpath, dirnames, filenames in os.walk(root) for filename in filenames]
    path = os.path.join(root, '.gitignore')
    try:
        with open(path) as f:
            spec = pathspec.PathSpec.from_lines('gitwildmatch', f.readlines() + ['.git/'])
        relpaths = [relpath for relpath in relpaths if not spec.match_file(relpath) or relpath == '.git/HEAD' or relpath.startswith('.git/refs/heads')]
    except FileNotFoundError:
        logging.warning(f"{path} not found")
    name = os.path.basename(root)
    path = os.path.join(dst, f"{name}.tar.gz")
    with tarfile.open(path, 'w:gz') as tar:
        for relpath in tqdm.tqdm(relpaths, desc=f'backup {name}'):
            tar.add(os.path.join(root, relpath), arcname=relpath)


class Fake(object):
    def __init__(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def register(self, *args, **kwargs):
        pass

    def put(self, record):
        pass

    def __call__(self, *args, **kwargs):
        pass


class Quit(object):
    pass


class Recorder(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.task = []

    def close(self):
        pass

    def register(self, timer, make):
        self.task.append((timer, make))

    def __call__(self, *args, **kwargs):
        for timer, make in self.task:
            if timer():
                try:
                    record = make(*args, **kwargs)
                    if callable(record):
                        self.put(record)
                except:
                    traceback.print_exc()

    def put(self, record):
        pass

    @staticmethod
    def new(**kwargs):
        if kwargs.get('recorder', None) is not None:
            return lamarckian.util.recorder.proxy.Client(**kwargs)
        elif kwargs.get('root', None) is not None:
            return lamarckian.util.recorder.Process(**kwargs)
        else:
            return lamarckian.util.recorder.Fake()


class Process(Recorder):
    class Worker(object):
        def __init__(self, queue, **kwargs):
            self.queue = queue
            self.kwargs = kwargs
            if glom.glom(kwargs['config'], 'recorder.evaluator', default=True) and 'ray' not in kwargs and 'index' not in kwargs:
                config = kwargs['config_test']
                Evaluator = lamarckian.evaluator.parse(*config['evaluator']['create'], config=config)
                self.evaluator = Evaluator(config=config, ray=lamarckian.util.ray_fake)
            root_log = os.path.join(kwargs['root'], 'log', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f'))
            os.makedirs(root_log, exist_ok=True)
            self.writer = torch.utils.tensorboard.SummaryWriter(root_log)
            texts = {key: oyaml.dump(kwargs[key], allow_unicode=True) for key in ('config', 'config_test')}
            for key, text in texts.items():
                with open(os.path.join(root_log, key + '.yml'), 'w') as f:
                    f.write(text)
            cmd = f"{sys.executable} {subprocess.list2cmdline(sys.argv)}"
            with open(os.path.join(root_log, 'cmd.txt'), 'w') as f:
                f.write(cmd)
            self.writer.add_text('cmd', cmd, self.kwargs.get('cost', 0))
            self.writer.add_text('config', texts['config'].replace('\n', '\n\n\t'), self.kwargs.get('cost', 0))
            self.writer.add_text('diff_config', '\n\n\t'.join(difflib.unified_diff(texts['config'].split('\n'), texts['config_test'].split('\n'))), self.kwargs.get('cost', 0))
            if 'index' not in kwargs:
                with open(os.path.join(root_log, 'platform.txt'), 'w') as f:
                    f.write(platform.platform())
                with open(os.path.join(root_log, 'python.txt'), 'w') as f:
                    f.write(platform.python_version())
                with open(os.path.join(root_log, 'env.tsv'), 'w') as f:
                    for key, value in os.environ.items():
                        f.write(f"{key}\t{value}\n")
                os.system(f"pip3 freeze > {os.path.join(root_log, 'requirements.txt')}")
                # backup codes
                names = set(glom.glom(kwargs['config'], 'recorder.backup', default=[]))
                for path in os.environ.get('PYTHONPATH', '').split(':'):
                    if os.path.basename(path) in names:
                        backup(os.path.expanduser(os.path.expandvars(path)), root_log)
                logging.info(f'tensorboard --logdir {root_log}')

        def close(self):
            if hasattr(self, 'evaluator'):
                self.evaluator.close()
            self.writer.close()

        def put(self, record):
            return self.queue.put(record)

        def __call__(self):
            record = self.queue.get()
            if callable(record):
                start = time.time()
                try:
                    record(self)
                except:
                    traceback.print_exc()
                finally:
                    duration = time.time() - start
                    if duration > 60:
                        logging.info(f"{record} finished in {humanfriendly.format_timespan(duration)}")
            return record

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        kwargs['config_test'] = lamarckian.util.config.make_test(copy.deepcopy(kwargs['config']))
        environ = os.environ.copy()
        for key, value in glom.glom(kwargs['config'], 'recorder.env', default={}).items():
            os.environ[key] = str(value)
        context = multiprocessing.get_context('spawn')
        self.queue = context.Queue(glom.glom(kwargs['config'], 'recorder.capacity'))
        self.process = context.Process(target=self.run, args=(self.queue,), kwargs=kwargs)
        self.process.start()
        for key in os.environ:
            del os.environ[key]
        for key, value in environ.items():
            os.environ[key] = value
        result = self.queue.get()
        if result is not None:
            e, s = result
            print(s)
            raise e

    def close(self):
        self.process.kill()
        self.process.join()

    def put(self, record):
        try:
            return self.queue.put(record, block=False)
        except queue.Full:
            pass  # traceback.print_exc()

    @staticmethod
    def run(queue, *args, **kwargs):
        try:
            import prctl
            prctl.set_pdeathsig(signal.SIGKILL)
        except ImportError:
            traceback.print_exc()
        setproctitle.setproctitle('recorder')
        logging.info(f"recorder CUDA devices: {torch.cuda.device_count()}")
        try:
            worker = Process.Worker(queue, *args, **kwargs)
            queue.put(None)
        except Exception as e:
            queue.put((e, traceback.format_exc()))
            raise
        while True:
            try:
                if isinstance(worker(), Quit):
                    break
            except KeyboardInterrupt:
                break
            except:
                traceback.print_exc()
        worker.close()


class Thread(Recorder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        kwargs['config_test'] = lamarckian.util.config.make_test(copy.deepcopy(kwargs['config']))
        self.queue = queue.Queue(glom.glom(kwargs['config'], 'recorder.capacity'))
        self.thread = threading.Thread(target=self.run, args=(self.queue,), kwargs=kwargs)
        self.thread.start()
        e = self.queue.get()
        if e is not None:
            raise e

    def close(self):
        self.queue.put(Quit())
        self.thread.join()

    def put(self, record):
        try:
            return self.queue.put(record, block=False)
        except queue.Full:
            traceback.print_exc()

    @staticmethod
    def run(queue, *args, **kwargs):
        import matplotlib
        matplotlib.use('Agg')
        asyncio.set_event_loop(asyncio.new_event_loop())
        return Process.run(queue, *args, **kwargs)


from . import proxy
