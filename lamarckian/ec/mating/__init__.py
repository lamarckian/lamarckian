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

import random
import operator
import copy

import numpy as np
import glom
import scipy.spatial.distance

import lamarckian


def tournament(population, competitors, compete=lambda *competitors: max(competitors, key=lambda item: item[1]['fitness']), random=random):
    competitors = random.sample(list(enumerate(population)), competitors)
    index, individual = compete(*competitors)
    return index, individual


def roulette_wheel(population, func, random=random):
    total = sum(map(func, population))
    end = random.uniform(0, total)
    seek = 0
    for i, individual in enumerate(population):
        seek += func(individual)
        if seek >= end:
            return i
    return -1


class Mating(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def close(self):
        pass


class Random(Mating):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.random = random.Random(glom.glom(kwargs['config'], 'seed', default=None))

    def __call__(self, population, choose):
        return self.random.sample(population, choose)


class Oldest(Mating):
    def __call__(self, population, choose):
        return sorted(population, key=operator.itemgetter('age'), reverse=True)[:choose]


class Tournament(Mating):
    def __init__(self, choose, **kwargs):
        super().__init__(**kwargs)
        assert choose > 1, choose
        self.choose = choose
        self.competitors = glom.glom(kwargs['config'], 'tournament.competitors')
        assert self.competitors >= 2
        self.compare = eval('lambda individual: ' + glom.glom(kwargs['config'], 'tournament.compare'))
        self.random = random.Random(glom.glom(kwargs['config'], 'seed', default=None))

    def compete(self, *competitors):
        return max(competitors, key=lambda item: self.compare(item[1]))

    def __call__(self, population):
        if len(population) > self.competitors:
            population = copy.copy(population)
            selected = []
            for _ in range(self.choose):
                index, individual = tournament(population, self.competitors, self.compete, random=self.random)
                population.pop(index)
                selected.append(individual)
            return selected
        else:
            return population


class TournamentNearest(Tournament):
    def __init__(self, choose, population, **kwargs):
        super().__init__(choose, **kwargs)
        objectives = np.array(list(map(operator.itemgetter('objective'), population)))
        try:
            import dmaps
            dist = dmaps.DistanceMatrix(objectives)
            dist.compute(metric=dmaps.metrics.euclidean)
            self.matrix = dist.get_distances()
        except ImportError:
            self.matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(objectives))

    def indices(self, dist, choose):
        return np.argpartition(dist, choose)[1:choose + 1]

    def __call__(self, population):
        index, individual = tournament(population, self.competitors, self.compete, random=self.random)
        indices = self.indices(self.matrix[index], self.choose - 1)
        neighbors = [population[i] for i in indices]
        return [individual] + neighbors


class TournamentFarthest(TournamentNearest):
    def indices(self, dist, choose):
        return np.argpartition(dist, choose)[-choose:]


class TournamentNearer(TournamentNearest):
    def score(self, dist):
        return max(dist) - dist

    def __call__(self, population):
        index, individual = tournament(population, self.competitors, self.compete, random=self.random)
        score = self.score(self.matrix[index])
        population = list(enumerate(score))
        population.pop(index)
        selected = [individual]
        for _ in range(self.choose - 1):
            i, _ = population.pop(roulette_wheel(population, operator.itemgetter(1)))
            selected.append(population[i])
        return selected


class TournamentFarther(TournamentNearer):
    def score(self, dist):
        return dist
