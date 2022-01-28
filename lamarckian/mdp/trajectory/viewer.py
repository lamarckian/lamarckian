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
import os
import argparse
import logging.config
import traceback

import torch
from PyQt5 import QtCore, QtWidgets

import lamarckian
from lamarckian.mdp.trajectory import widget


class Widget(QtWidgets.QDialog):
    def __init__(self, mdp, data, **kwargs):
        super().__init__()
        self.mdp = mdp
        self.data = data
        self.kwargs = kwargs
        me = mdp.get('me', 0)
        self.setWindowFlags(QtCore.Qt.Window)
        # widget
        layout = QtWidgets.QVBoxLayout(self)
        self.widget_action = widget.Action(self, mdp['encoding'], data['trajectory'], me=me, **kwargs)
        self.widget_action.setFocusPolicy(QtCore.Qt.NoFocus)
        layout.addWidget(self.widget_action)
        self.widget_reward = widget.Reward(self, mdp['encoding'], me=me, **kwargs)
        self.widget_reward.plot(data['trajectory'])
        layout.addWidget(self.widget_reward)
        self.widget_state = widget.State(self, mdp['encoding'], me=me, **data, **kwargs)
        layout.addWidget(self.widget_state)
        self.widget_frame = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.widget_frame.setMaximum(len(data['trajectory']) - 1)
        layout.addWidget(self.widget_frame)
        self.widget_frame.setFocus()
        self.widget_frame.valueChanged.connect(self.__call__)
        self.widget_action.cellClicked.connect(lambda row, column: self.widget_frame.setValue(column))
        for i in range(self.widget_reward.count()):
            self.widget_reward.widget(i).figure.canvas.mpl_connect('button_press_event', self.on_click)
        self(0)

    def closeEvent(self, event):
        self.widget_state.close()
        return super().closeEvent(event)

    def __call__(self, frame):
        trajectory = self.data['trajectory']
        self.widget_state(trajectory[frame], **(trajectory[frame + 1] if frame < len(trajectory) - 1 else self.data['exp']))
        self.widget_action.selectColumn(frame)
        self.widget_reward(frame)

    def on_click(self, event):
        if event.xdata is not None:
            frame = int(event.xdata)
            if 0 <= frame < len(self.data['trajectory']):
                self.widget_frame.setValue(frame)


def main():
    args = make_args()
    config = {}
    for path in sum(args.config, []):
        config = lamarckian.util.config.read(path, config)
    for cmd in sum(args.modify, []):
        lamarckian.util.config.modify(config, cmd)
    logging.config.dictConfig(config['logging'])
    app = QtWidgets.QApplication(sys.argv)
    root = os.path.expanduser(os.path.expandvars(config['root']))
    root = QtWidgets.QFileDialog.getExistingDirectory(caption='choose the rollout directory', directory=root)
    if root:
        mdp = torch.load(root + '.mdp.pth')
        mdp['me'] = mdp.get('me', 0)
        for index in sorted(list(map(int, filter(str.isdigit, {name.split('.')[0] for name in os.listdir(root)})))):
            prefix = os.path.join(root, str(index))
            logging.info(prefix)
            data = torch.load(f'{prefix}.pth')
            widget = Widget(mdp, data, config=config)
            try:
                widget.setWindowTitle(f"{prefix}: {data['result']}")
            except KeyError:
                widget.setWindowTitle(prefix)
                traceback.print_exc()
            widget.show()
            app.exec_()


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', action='append', default=[[os.path.dirname(os.path.abspath(lamarckian.__file__)) + '.yml']])
    parser.add_argument('-m', '--modify', nargs='+', action='append', default=[])
    return parser.parse_args()


if __name__ == '__main__':
    main()
