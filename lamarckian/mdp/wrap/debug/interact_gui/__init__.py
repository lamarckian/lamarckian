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

import sys
import inspect
import functools
import threading

import torch
import glom
import zmq
import msgpack
from PyQt5 import QtCore, QtWidgets

import lamarckian
from lamarckian.mdp.trajectory import widget


def slice(role, **kwargs):
    for key in 'discrete continuous'.split():
        if key in kwargs:
            kwargs[key] = [torch.moveaxis(action, -1, 0)[role] for action in kwargs[key]]
    for key in 'error reward'.split():
        kwargs[key] = kwargs[key][role]
    return kwargs


class WidgetRole(QtWidgets.QWidget):
    def __init__(self, parent, role, encoding, **kwargs):
        super().__init__()
        self.parent = parent
        self.role = role
        self.encoding = encoding
        self.kwargs = kwargs
        layout = QtWidgets.QVBoxLayout(self)
        self.widget_action = widget.Action(self, encoding, **kwargs)
        self.widget_action.setFocusPolicy(QtCore.Qt.NoFocus)
        layout.addWidget(self.widget_action)
        layout.addWidget(self.create_widgets1(role))
        for i in range(self.widget_reward.count()):
            self.widget_reward.widget(i).figure.canvas.mpl_connect('button_press_event', lambda event: None if event.xdata is None else self.on_frame(int(event.xdata)))

    def create_widgets1(self, role):
        me = self.kwargs.get('me', 0)
        model = self.encoding['blob']['models'][me]
        names = model['discrete'][0]
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.widget_pred = QtWidgets.QTableWidget()
        self.widget_pred.setRowCount(len(names))
        self.widget_pred.verticalHeader().hide()
        self.widget_pred.setColumnCount(1)
        self.widget_pred.horizontalHeader().hide()
        keys = glom.glom(self.kwargs['config'], 'mdp.debug.interact.shortcut', default=[])
        width = 0
        for action, (name, key) in enumerate(zip(names, keys + [None] * (len(names) - len(keys)))):
            cell = QtWidgets.QPushButton(name)
            cell.clicked.connect(functools.partial(self.set_action, role, action))
            cell.setShortcut(str(key))
            self.widget_pred.setCellWidget(action, 0, cell)
            width = max(cell.width(), width)
        self.widget_pred.setFixedWidth(width)
        # self.widget_pred.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        splitter.addWidget(self.widget_pred)
        self.widget_reward = widget.Reward(self, self.encoding, self.widget_action.trajectory, **self.kwargs)
        # self.widget_reward.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        splitter.addWidget(self.widget_reward)
        return splitter

    def set_action(self, role, action):
        self.parent.action['role'] = role
        torch.moveaxis(self.parent.action['discrete'][0], -1, 0)[role] = action
        self.parent.event_action.set()

    def on_frame(self, frame):
        trajectory = self.widget_action.trajectory
        exp = trajectory[frame]
        try:
            exp_ = trajectory[frame + 1]
        except IndexError:
            exp_ = dict(state=self.parent.state_)
        self.parent.widget_state(exp, **exp_)
        self.widget_action.selectColumn(frame)
        if len(self.widget_reward) == len(trajectory):
            self.widget_reward(frame)


class Widget(QtWidgets.QWidget):
    signal_reset = QtCore.pyqtSignal(dict)
    signal_append = QtCore.pyqtSignal(dict, dict)
    signal_set_default = QtCore.pyqtSignal(int, dict)

    def __init__(self, encoding, **kwargs):
        super().__init__()
        self.encoding = encoding
        self.kwargs = kwargs
        self.event_action = threading.Event()
        layout = QtWidgets.QVBoxLayout(self)
        self.widget_roles = QtWidgets.QTabWidget()
        for role in range(self.encoding['blob'].get('roles', 1)):
            self.widget_roles.addTab(WidgetRole(self, role, encoding, **kwargs), str(role))
        layout.addWidget(self.widget_roles)
        self.widget_state = widget.State(self, encoding, **kwargs)
        layout.addWidget(self.widget_state)
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            signal = getattr(self, f"signal_{name}", None)
            if signal is not None:
                signal.connect(method)

    def reset(self, state):
        exp = dict(state=state)
        for role in range(self.widget_roles.count()):
            widget = self.widget_roles.widget(role)
            widget.widget_action.set([])
        self.widget_state.guess(exp)
        self.widget_state(exp, state=state)

    def append(self, exp, state_):
        self.state_ = state_
        for role in range(self.widget_roles.count()):
            widget = self.widget_roles.widget(role)
            widget.widget_action.append(slice(role, **{key: value for key, value in exp.items() if key not in {'role'}}))
        widget = self.widget_roles.currentWidget()
        widget.widget_reward.plot()
        widget.on_frame(len(widget.widget_action) - 1)

    def set_default(self, frame, action):
        self.action = action
        for role, discrete in enumerate(torch.unbind(action['discrete'][0], -1)):
            discrete = discrete.item()
            widget = self.widget_roles.widget(role)
            for i in range(widget.widget_pred.rowCount()):
                cell = widget.widget_pred.cellWidget(i, 0)
                cell.setDefault(discrete == i)
            cell = widget.widget_pred.cellWidget(discrete, 0)
            self.widget_roles.setTabText(role, cell.text())
        self.setWindowTitle(str(frame))

    def __call__(self, frame, **kwargs):
        self.signal_set_default.emit(frame, kwargs)
        self.event_action.clear()
        self.event_action.wait()
        return self.action


def loop(widget, socket):
    while True:
        name, args, kwargs = msgpack.loads(socket.recv(), object_hook=lamarckian.util.serialize.decode, strict_map_key=False)
        if name is None:
            socket.send(msgpack.dumps(widget.close(), default=lamarckian.util.serialize.encode))
            break
        try:
            signal = getattr(widget, f"signal_{name}")
            assert not kwargs, (name, kwargs.keys())
            result = signal.emit(*args, **kwargs)
        except AttributeError:
            method = getattr(widget, name)
            result = method(*args, **kwargs)
        socket.send(msgpack.dumps(result, default=lamarckian.util.serialize.encode))


def run(encoding, **kwargs):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.connect(f"tcp://{kwargs['host']}:{kwargs['port']}")
    app = QtWidgets.QApplication(sys.argv)
    widget = Widget(encoding, **kwargs)
    widget.show()
    thread = threading.Thread(target=functools.partial(loop, widget, socket))
    thread.start()
    app.exec_()
    thread.join()
