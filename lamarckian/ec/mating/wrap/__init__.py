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


def best_blob(mating):
    class Mating(mating):
        def __call__(self, population, choose):
            ancestor = super().__call__(population, choose)
            blob = max(population, key=lambda individual: individual['result']['fitness'])['decision']['blob']
            for individual in ancestor:
                individual['decision']['blob'] = blob
            return ancestor
    return Mating
