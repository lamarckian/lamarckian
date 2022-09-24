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

import itertools
import numbers
import traceback

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5 import QtCore, QtWidgets, QtGui

from . import state


def get_action_color(n):
    return [prop['color'] for _, prop in zip(range(n), itertools.cycle(plt.rcParams['axes.prop_cycle']))]


class State(QtWidgets.QLineEdit):
    def __init__(self, parent, encoding, **kwargs):
        super().__init__()
        self.parent = parent
        self.encoding = encoding
        self.kwargs = kwargs
        self.installEventFilter(self)
        self.widgets = []
        if 'trajectory' in kwargs:
            self.guess(kwargs['trajectory'][0])
        elif 'exp' in kwargs:
            self.guess(kwargs['exp'])

    def closeEvent(self, event):
        for widget in self.widgets:
            widget.close()
        return super().closeEvent(event)

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.KeyPress and obj is self:
            if event.key() == QtCore.Qt.Key_Return and self.hasFocus():
                self.apply()
        return super().eventFilter(obj, event)

    def guess(self, exp):
        comp = [f"state.inputs.{i}:{encoding.get('viewer', f'Feature{len(input.shape) - 1}')}" for i, (encoding, input) in enumerate(zip(self.encoding['blob']['models'][0]['inputs'], exp['state']['inputs']))]
        for key, value in exp['state'].items():
            if key == 'image':
                comp.append(f'state.{key}:Image')
            elif key == 'legal':
                comp.append(f'state.{key}:Legal')
            elif key == 'probs':
                comp.append(f'state.{key}:Probs')
        self.setText('; '.join(comp))
        self.apply()

    def apply(self):
        def create(s):
            select, widget = s.split(':')
            cls = eval(f'state.{widget}')
            return cls(self, self.encoding, select, **self.kwargs)
        for widget in self.widgets:
            widget.close()
        self.widgets = list(map(create, filter(None, map(str.strip, self.text().split(';')))))
        for widget in self.widgets:
            try:
                widget.show()
                if callable(widget) and hasattr(self, 'exp'):
                    args, kwargs = self._
                    widget(*args, **kwargs)
            except:
                traceback.print_exc()

    def __call__(self, *args, **kwargs):
        for widget in self.widgets:
            try:
                if callable(widget):
                    widget(*args, **kwargs)
            except:
                traceback.print_exc()
        self._ = (args, kwargs)


class Reward(QtWidgets.QTabWidget):
    def __init__(self, parent, encoding, trajectory, **kwargs):
        super().__init__()
        self.parent = parent
        self.encoding = encoding
        self.trajectory = trajectory
        self.kwargs = kwargs
        for reward in encoding['blob']['reward']:
            widget = FigureCanvasQTAgg(plt.figure())
            self.addTab(widget, reward)
        model = encoding['blob']['models'][kwargs.get('me', 0)]
        names = model['discrete'][0]
        self.color = get_action_color(len(names))
        self.currentChanged.connect(self.plot)

    def closeEvent(self, event):
        plt.close(self.figure)

    def plot(self):
        x = np.arange(len(self.trajectory))
        index = self.currentIndex()
        y = np.array([exp['reward'][index] for exp in self.trajectory])
        widget = self.currentWidget()
        ax = widget.figure.gca()
        ax.cla()
        ax.plot(x, y)
        widget.figure.tight_layout()
        widget.draw()
        self.artist = []

    def __call__(self, frame):
        try:
            for artist in self.artist:
                artist.remove()
        except AttributeError:
            pass
        self.artist = []
        exp = self.trajectory[frame]
        action = int(exp['discrete'][0])
        color = self.color[action]
        marker = '-' if self.trajectory[frame].get('success', True) else '--'
        reward = self.trajectory[frame]['reward'][self.currentIndex()]
        widget = self.currentWidget()
        ax = widget.figure.gca()
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        self.artist.append(ax.plot([xlim[0], frame], [reward, reward], marker, c=color)[0])
        self.artist.append(ax.plot([frame, frame], ylim, marker, c=color)[0])
        self.artist.append(ax.text(xlim[0], reward, f"{reward:.1f}"))
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        widget.draw()

    def __len__(self):
        return len(self.trajectory)


class Action(QtWidgets.QTableWidget):
    def __init__(self, parent, encoding, trajectory=[], **kwargs):
        super().__init__()
        self.parent = parent
        self.encoding = encoding
        self.trajectory = trajectory
        self.kwargs = kwargs
        model = encoding['blob']['models'][kwargs.get('me', 0)]
        self.names = model['discrete'][0]
        self.color = [QtGui.QColor(*np.array(matplotlib.colors.to_rgb(color)) * 255) for color in get_action_color(len(self.names))]
        rows = ['skill', 'error', 'message', 'message_']
        self.setRowCount(len(rows))
        self.setVerticalHeaderLabels(rows)
        self.verticalHeaderItem(0).setToolTip(' '.join(self.names))
        self.set(trajectory)
        self.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

    def set_exp(self, j, exp):
        action = int(exp['discrete'][0])
        item = QtWidgets.QTableWidgetItem(self.names[action])
        item.setBackground(self.color[action])
        item.setToolTip(str(action))
        self.setItem(0, j, item)
        error = exp.get('error', '')
        if error:
            self.setItem(1, j, QtWidgets.QTableWidgetItem(error))
        self.setItem(2, j, QtWidgets.QTableWidgetItem())
        self.setItem(3, j, QtWidgets.QTableWidgetItem())

    def set_message(self, j, message):
        item = self.item(2, j)
        item.setText(message)
        item.setToolTip(message)
        try:
            message_ = self.trajectory[j + 1].get('message', '')
            item = self.item(3, j)
            item.setText(message_)
            item.setToolTip(message_)
        except IndexError:
            pass

    def set(self, trajectory):
        self.trajectory[:] = trajectory
        size = len(trajectory)
        self.setColumnCount(size)
        self.setHorizontalHeaderLabels(list(map(str, range(size))))
        for j, exp in enumerate(trajectory):
            self.set_exp(j, exp)
        self.resizeColumnsToContents()
        for j, exp in enumerate(trajectory):
            self.set_message(j, exp.get('message', ''))

    def append(self, exp):
        self.trajectory.append(exp)
        size = len(self.trajectory)
        self.setColumnCount(size)
        self.setHorizontalHeaderLabels(list(map(str, range(size))))
        j = size - 1
        self.set_exp(j, exp)
        self.resizeColumnToContents(j)
        self.set_message(j, exp.get('message', ''))

    def __len__(self):
        return len(self.trajectory)
