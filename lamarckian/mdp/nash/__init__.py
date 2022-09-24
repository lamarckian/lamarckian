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

import numpy as np


def fictitious_play(payoff, iteration=10000, seed=None):
    rs = np.random.RandomState(seed)
    size, = set(payoff.shape)
    counts = np.zeros(size, np.int)
    for i in range(iteration):
        scores = payoff @ counts
        bests, = np.nonzero(scores == np.max(scores))
        best = rs.choice(bests)
        counts[best] += 1
    return counts


def to_probs(counts, threshold=0.01):
    assert 0 <= threshold < 1, threshold
    probs = counts / counts.sum()
    if threshold > 0:
        value = 0
        for i in np.argsort(probs):
            value += probs[i]
            if value < threshold:
                probs[i] = 0
            else:
                break
        probs /= probs.sum()
    return probs


def clustering(payoff, get_nash=lambda payoff: to_probs(fictitious_play(payoff))):
    size, = set(payoff.shape)
    assert size > 1, size
    indexes = np.array(list(range(size)))
    while len(payoff) > 0:
        if len(payoff) > 1:
            probs = get_nash(payoff)
            mask = probs > 0
            yield indexes[mask], probs[mask]
            mask = ~mask
            payoff, indexes = payoff[mask][:, mask], indexes[mask]
        else:
            yield indexes, np.ones([1])
            break
