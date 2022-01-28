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

import types
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import seaborn as sns
from PyQt5 import QtWidgets, QtGui, QtCore
import glom


class Text(QtWidgets.QWidget):
    def __init__(self, parent, encoding, spec, **kwargs):
        super().__init__()
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        self.parent = parent
        self.encoding = encoding
        self.spec = spec
        self.kwargs = kwargs
        layout = QtWidgets.QVBoxLayout(self)
        self.widget_text = QtWidgets.QTextEdit()
        layout.addWidget(self.widget_text)
        self.widget_text_ = QtWidgets.QTextEdit()
        layout.addWidget(self.widget_text_)
        self.setWindowTitle(spec)

    def __call__(self, exp, **kwargs):
        state = glom.glom(exp, self.spec)
        assert isinstance(state, str), type(state)
        self.widget_text.setText(state)
        if kwargs:
            state_ = glom.glom(kwargs, self.spec)
            assert isinstance(state_, str), type(state)
            self.widget_text_.setText(state_)


class Feature1(QtWidgets.QWidget):
    def __init__(self, parent, encoding, spec, **kwargs):
        super().__init__()
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        self.parent = parent
        self.encoding = encoding
        self.spec = spec
        self.kwargs = kwargs
        header, = self.get_header()
        layout = QtWidgets.QVBoxLayout(self)
        self.widget_search = QtWidgets.QLineEdit()
        self.widget_search.textChanged.connect(self.on_search)
        layout.addWidget(self.widget_search)
        self.widget_state = QtWidgets.QTableWidget()
        self.widget_state.setRowCount(2)
        self.widget_state.verticalHeader().hide()
        self.widget_state.setColumnCount(len(header))
        self.widget_state.setHorizontalHeaderLabels(header)
        for i in range(self.widget_state.rowCount()):
            for j in range(self.widget_state.columnCount()):
                self.widget_state.setItem(i, j, QtWidgets.QTableWidgetItem())
        self.widget_state.resizeColumnsToContents()
        self.widget_state.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.widget_state.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.background = types.SimpleNamespace(item=self.widget_state.item(0, 0).background())
        layout.addWidget(self.widget_state)
        self.setWindowTitle(spec)
        self.cmap = self.get_cmap()

    def get_header(self):
        header = self.kwargs.get('header', None)
        if header is not None:
            return header
        elif 'dim' in self.kwargs:
            return [list(map(str, range(self.kwargs['dim'])))]
        else:
            index = int(re.match(r'state\.inputs\.(\d+)', self.spec).group(1))
            me = self.kwargs.get('me', 0)
            input = self.encoding['blob']['models'][me]['inputs'][index]
            header = input.get('header', None)
            if header is None:
                dim, = input['shape']
                return [list(map(str, range(dim)))]
            else:
                return header

    def get_cmap(self):
        if 'trajectory' in self.kwargs:
            trajectory = np.array([glom.glom(exp, self.spec) for exp in self.kwargs['trajectory']])
            _, dim = trajectory.shape
            assert dim == self.widget_state.columnCount(), (dim, self.widget_state.columnCount())
            vmin, vmax = trajectory.min(0), trajectory.max(0)
        else:
            dim = self.widget_state.columnCount()
            vmin, vmax = np.full(dim, np.finfo(float).max), np.full(dim, np.finfo(float).min)
        return [matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin, vmax), cmap=matplotlib.cm.jet) for vmin, vmax in zip(vmin, vmax)]

    def update_cmap(self, state):
        for i, (value, cmap) in enumerate(zip(state, self.cmap)):
            cmap.norm.vmin, cmap.norm.vmax = min(cmap.norm.vmin, value), max(cmap.norm.vmax, value)
            self.widget_state.horizontalHeaderItem(i).setToolTip(f'{i}: {cmap.norm.vmin}--{cmap.norm.vmax}')

    @staticmethod
    def set_value(item, value, cmap):
        item.setText(str(value))
        if cmap.norm.vmin < cmap.norm.vmax:
            color = np.array(cmap.to_rgba(value)[:3]) * 255
            item.setBackground(QtGui.QColor(*color))

    def __call__(self, exp, **kwargs):
        state = glom.glom(exp, self.spec)
        dim, = state.shape
        assert dim == self.widget_state.columnCount(), (dim, self.widget_state.columnCount())
        self.update_cmap(state)
        if kwargs:
            state_ = glom.glom(kwargs, self.spec)
            dim, = state_.shape
            assert dim == self.widget_state.columnCount(), (dim, self.widget_state.columnCount())
            self.update_cmap(state_)
        for j, value in enumerate(state):
            self.set_value(self.widget_state.item(0, j), value, self.cmap[j])
        if kwargs:
            for j, value in enumerate(state_):
                self.set_value(self.widget_state.item(1, j), value, self.cmap[j])

    def on_search(self, text):
        for j in range(self.widget_state.columnCount()):
            self.widget_state.item(0, j).setSelected(False)
        if text:
            for j in range(self.widget_state.columnCount()):
                if self.widget_state.horizontalHeaderItem(j).text().startswith(text):
                    item = self.widget_state.item(0, j)
                    self.widget_state.scrollToItem(item)
                    item.setSelected(True)
                    return


class Feature2(FigureCanvasQTAgg):
    def __init__(self, parent, encoding, spec, **kwargs):
        super().__init__(plt.figure())
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        self.parent = parent
        self.encoding = encoding
        self.spec = spec
        self.kwargs = kwargs
        index = int(re.match(r'state\.inputs\.(\d+)', self.spec).group(1))
        me = self.kwargs.get('me', 0)
        input = self.encoding['blob']['models'][me]['inputs'][index]
        try:
            self.header, _ = input['header']
        except (KeyError, TypeError):
            self.header = False
        self.setWindowTitle(spec)
        self.figure.tight_layout()

    def closeEvent(self, event):
        plt.close(self.figure)

    def __call__(self, exp, **kwargs):
        state = glom.glom(kwargs, self.spec)
        assert len(state.shape) == 2, state.shape
        ax = self.figure.gca()
        ax.cla()
        sns.heatmap(state, cbar=False, xticklabels=False, yticklabels=self.header, square=True, ax=ax, **glom.glom(self.kwargs['config'], 'mdp.debug.feature2', default={}))
        self.draw()


class Feature3(QtWidgets.QWidget):
    def __init__(self, parent, encoding, spec, **kwargs):
        super().__init__()
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        self.parent = parent
        self.encoding = encoding
        self.spec = spec
        self.kwargs = kwargs
        layout = QtWidgets.QVBoxLayout(self)
        self.widget_channel = QtWidgets.QSpinBox()
        self.widget_channel.setMinimum(0)
        self.widget_channel.setMaximum(0)
        self.widget_channel.valueChanged.connect(self.__call__)
        layout.addWidget(self.widget_channel)
        self.widget_image = FigureCanvasQTAgg(plt.figure())
        layout.addWidget(self.widget_image)
        self.setWindowTitle(spec)
        self.widget_image.figure.tight_layout()

    def closeEvent(self, event):
        plt.close(self.widget_image.figure)

    def __call__(self, exp, **kwargs):
        state = glom.glom(kwargs, self.spec)
        assert len(state.shape) == 3, state.shape
        if self.widget_channel.maximum() <= 0:
            self.widget_channel.setMaximum(len(state) - 1)
        channel = self.widget_channel.value()
        image = state[channel]
        ax = self.widget_image.figure.gca()
        ax.cla()
        ax.imshow(image)
        self.widget_image.draw()


class Image(QtWidgets.QLabel):
    def __init__(self, parent, encoding, spec, **kwargs):
        super().__init__()
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        self.parent = parent
        self.encoding = encoding
        self.spec = spec
        self.kwargs = kwargs
        self.setPixmap(QtGui.QPixmap())
        self.setWindowTitle(spec)

    def __call__(self, exp, **kwargs):
        state = glom.glom(kwargs, self.spec)
        self.pixmap().loadFromData(state)
        self.repaint()
        self.update()
        pixmap = self.pixmap()
        self.setFixedSize(pixmap.width(), pixmap.height())


class Legal(QtWidgets.QWidget):
    def __init__(self, parent, encoding, spec, **kwargs):
        super().__init__()
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        self.parent = parent
        self.encoding = encoding
        self.spec = spec
        self.kwargs = kwargs
        header = self.get_header()
        layout = QtWidgets.QVBoxLayout(self)
        self.widget_state = QtWidgets.QTableWidget()
        self.widget_state.setRowCount(2)
        self.widget_state.verticalHeader().hide()
        self.widget_state.setColumnCount(len(header))
        self.widget_state.setHorizontalHeaderLabels(header)
        for i in range(self.widget_state.rowCount()):
            for j in range(self.widget_state.columnCount()):
                cell = QtWidgets.QCheckBox()
                cell.setEnabled(False)
                self.widget_state.setCellWidget(i, j, cell)
        self.widget_state.resizeColumnsToContents()
        self.widget_state.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.widget_state.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        layout.addWidget(self.widget_state)
        self.setWindowTitle(spec)

    def get_header(self):
        me = self.kwargs.get('me', 0)
        names, = self.encoding['blob']['models'][me]['discrete']
        return names

    def __call__(self, exp, **kwargs):
        state = glom.glom(exp, self.spec)
        dim, = state.shape
        assert dim == self.widget_state.columnCount(), (dim, self.widget_state.columnCount())
        for j, value in enumerate(state):
            self.widget_state.cellWidget(0, j).setCheckState(int(value) * 2)
        if kwargs:
            state_ = glom.glom(kwargs, self.spec)
            dim, = state_.shape
            assert dim == self.widget_state.columnCount(), (dim, self.widget_state.columnCount())
            for j, value in enumerate(state_):
                self.widget_state.cellWidget(1, j).setCheckState(int(value) * 2)


class Probs(FigureCanvasQTAgg):
    def __init__(self, parent, encoding, spec, **kwargs):
        super().__init__(plt.figure())
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        self.parent = parent
        self.encoding = encoding
        self.spec = spec
        self.kwargs = kwargs
        self.setWindowTitle(spec)
        self.figure.tight_layout()

    def closeEvent(self, event):
        plt.close(self.figure)

    def __call__(self, exp, **kwargs):
        state = glom.glom(exp, self.spec)
        assert len(state.shape) == 1, state.shape
        ax = self.figure.gca()
        ax.cla()
        x = np.arange(len(state))
        ax.bar(x, state)
        ax.set_xticks(x)
        ax.set_xticklabels(self.parent.parent.widget_action.header)
        self.figure.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
        self.draw()
