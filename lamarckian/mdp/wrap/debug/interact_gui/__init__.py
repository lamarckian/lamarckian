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

import sys
import inspect
import functools
import threading

import glom
import zmq
import msgpack
from PyQt5 import QtCore, QtWidgets

import lamarckian
from lamarckian.mdp.trajectory import widget


class Widget(QtWidgets.QDialog):
    signal_reset = QtCore.pyqtSignal(dict)
    signal_append = QtCore.pyqtSignal(dict, dict)
    signal_set_default = QtCore.pyqtSignal(int, int)

    def __init__(self, encoding, **kwargs):
        super().__init__()
        self.encoding = encoding
        self.kwargs = kwargs
        self.event_action = threading.Event()
        layout = QtWidgets.QVBoxLayout(self)
        self.widget_action = widget.Action(self, encoding, **kwargs)
        self.widget_action.setFocusPolicy(QtCore.Qt.NoFocus)
        layout.addWidget(self.widget_action)
        layout.addLayout(self.create_layout1())
        self.widget_state = widget.State(self, encoding, **kwargs)
        layout.addWidget(self.widget_state)
        self.widget_frame = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.widget_frame.setMinimum(0)
        self.widget_frame.setMaximum(0)
        layout.addWidget(self.widget_frame)
        self.widget_frame.setFocus()
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            signal = getattr(self, f"signal_{name}", None)
            if signal is not None:
                signal.connect(method)
        self.widget_frame.valueChanged.connect(self.on_frame)
        self.widget_action.cellClicked.connect(lambda row, column: self.widget_frame.setValue(column))
        for i in range(self.widget_reward.count()):
            self.widget_reward.widget(i).figure.canvas.mpl_connect('button_press_event', self.on_click)

    def create_layout1(self):
        me = self.kwargs.get('me', 0)
        model = self.encoding['blob']['models'][me]
        names = model['discrete'][0]
        layout = QtWidgets.QHBoxLayout()
        self.widget_pred = QtWidgets.QTableWidget()
        self.widget_pred.setRowCount(len(names))
        self.widget_pred.verticalHeader().hide()
        self.widget_pred.setColumnCount(1)
        self.widget_pred.horizontalHeader().hide()
        keys = glom.glom(self.kwargs['config'], 'mdp.debug.interact.shortcut', default=[])
        for i, (name, key) in enumerate(zip(names, keys + [None] * (len(names) - len(keys)))):
            cell = QtWidgets.QPushButton(name)
            cell.clicked.connect(functools.partial(self.set_action, i))
            cell.setShortcut(str(key))
            self.widget_pred.setCellWidget(i, 0, cell)
        self.widget_pred.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        layout.addWidget(self.widget_pred)
        self.widget_reward = widget.Reward(self, self.encoding, **self.kwargs)
        # self.widget_reward.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        layout.addWidget(self.widget_reward)
        return layout

    def reset(self, state):
        self.widget_action.set([])
        self.widget_frame.setMaximum(0)
        exp = dict(state=state)
        self.widget_state.guess(exp)
        self.widget_state(exp, state=state)

    def append(self, exp, state_):
        self.widget_action.append(exp)
        self.state_ = state_
        value = len(self.widget_action) - 1
        self.widget_frame.setMaximum(value)
        self.widget_frame.setValue(value)
        self.widget_reward.plot(self.widget_action.trajectory)

    def set_default(self, frame, action):
        assert 0 <= action < self.widget_pred.rowCount(), (action, self.widget_pred.rowCount())
        for i in range(self.widget_pred.rowCount()):
            cell = self.widget_pred.cellWidget(i, 0)
            cell.setDefault(False)
        cell = self.widget_pred.cellWidget(action, 0)
        cell.setDefault(True)
        cell.setFocus()
        self.setWindowTitle(f"{frame}: {cell.text()}")

    def set_action(self, action):
        for i in range(self.widget_pred.rowCount()):
            cell = self.widget_pred.cellWidget(i, 0)
            if cell.isDefault():
                cell.setFocus()
                break
        self.action = action
        self.event_action.set()

    def __call__(self, frame, action):
        self.signal_set_default.emit(frame, action)
        self.event_action.clear()
        self.event_action.wait()
        return self.action

    def on_frame(self, frame):
        trajectory = self.widget_action.trajectory
        exp = trajectory[frame]
        try:
            exp_ = trajectory[frame + 1]
        except IndexError:
            exp_ = dict(state=self.state_)
        self.widget_state(exp, **exp_)
        self.widget_action.selectColumn(frame)
        if len(self.widget_reward) == len(trajectory):
            self.widget_reward(frame)

    def on_click(self, event):
        if event.xdata is not None:
            frame = int(event.xdata)
            if 0 <= frame < len(self.widget_action.trajectory):
                self.widget_frame.setValue(frame)


def loop(widget, socket):
    while True:
        name, args, kwargs = msgpack.loads(socket.recv(), object_hook=lamarckian.util.rpc.util.decode, strict_map_key=False)
        if name is None:
            socket.send(msgpack.dumps(widget.close(), default=lamarckian.util.rpc.util.encode))
            break
        try:
            signal = getattr(widget, f"signal_{name}")
            assert not kwargs, (name, kwargs.keys())
            result = signal.emit(*args, **kwargs)
        except AttributeError:
            method = getattr(widget, name)
            result = method(*args, **kwargs)
        socket.send(msgpack.dumps(result, default=lamarckian.util.rpc.util.encode))


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
