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
import logging.config

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

import lamarckian


def main():
    args = make_args()
    config = {}
    for path in sum(args.config, []):
        config = lamarckian.util.config.read(path, config)
    for cmd in sum(args.modify, []):
        lamarckian.util.config.modify(config, cmd)
    logging.config.dictConfig(config['logging'])
    root = os.path.expanduser(os.path.expandvars(config['root']))
    # payoff
    payoff = torch.load(os.path.join(root, 'payoff.pth'))
    if np.isnan(payoff).any():
        label = False
    else:
        probs = lamarckian.mdp.nash.to_probs(lamarckian.mdp.nash.fictitious_play(payoff))
        label = [format(prob, '.2f') if prob > 0 else ' ' * 4 for prob in probs]
    fig = plt.figure()
    ax = fig.gca()
    limit = np.abs(payoff).max()
    sns.heatmap(payoff, vmin=-limit, vmax=limit, cmap=plt.get_cmap('seismic'), annot=args.annot, cbar=False, xticklabels=False, yticklabels=label, square=True, ax=ax)
    fig.tight_layout()
    plt.show()


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', action='append', default=[[os.path.dirname(lamarckian.__file__) + '.yml']])
    parser.add_argument('-m', '--modify', nargs='+', action='append', default=[])
    parser.add_argument('-a', '--annot', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main()
