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

import numbers

import numpy as np
import glom


def get_hparam(encoding, population, tag='hparam'):
    encoding = {coding: encoding for coding, encoding in encoding.items() if coding in {'real', 'integer'}}
    return {f"{tag}/{coding}/{header}": np.array([individual['decision'][coding][i] for individual in population]).T for coding, encoding in encoding.items() for i, header in enumerate(encoding)}


def get_population(population, tag='population'):
    results = [individual['result'] for individual in population]
    keys = {key for result in results for key, value in result.items() if not key.startswith('_') and isinstance(value, (numbers.Integral, numbers.Real))}
    return {
        **{f"{tag}/result/{key}": np.array([result[key] for result in results if key in result]) for key in keys},
        **{f"{tag}/cost/{key}": np.array([glom.glom(individual, f"cost.{key}", default=np.nan) / (sum(individual.get('cost', {}).values()) + np.finfo(float).eps) for individual in population]) for key in {key for individual in population for key in individual.get('cost', {})}},
        **{f"{tag}/duration/{key}": np.array([glom.glom(individual, f"duration.{key}", default=np.nan) / (sum(individual.get('duration', {}).values()) + np.finfo(float).eps) for individual in population]) for key in {key for individual in population for key in individual.get('duration', {})}},
    }


def get_population_minmax(population, tag='population'):
    results = [individual['result'] for individual in population]
    keys = {key for result in results for key, value in result.items() if not key.startswith('_') and isinstance(value, (numbers.Integral, numbers.Real))}
    return {
        **{f"{tag}/result_min/{key}": np.min([result[key] for result in results if key in result]) for key in keys},
        **{f"{tag}/result_max/{key}": np.max([result[key] for result in results if key in result]) for key in keys},
        **{
            f"{tag}/results_min": min(len(individual['results']) for individual in population),
            f"{tag}/results_max": max(len(individual['results']) for individual in population),
        },
    }
