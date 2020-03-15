"""
Regularized evolution as described in:
Real, E., Aggarwal, A., Huang, Y., and Le, Q. V.
Regularized Evolution for Image Classifier Architecture Search.
In Proceedings of the Conference on Artificial Intelligence (AAAI’19)

The code is based one the original regularized evolution open-source implementation:
https://colab.research.google.com/github/google-research/google-research/blob/master/evolution/regularized_evolution_algorithm/regularized_evolution.ipynb

NOTE: This script has certain deviations from the original code owing to the search space of the benchmarks used:
1) The fitness function is not accuracy but error and hence the negative error is being maximized.
2) The architecture is a ConfigSpace object that defines the model architecture parameters.

"""

import argparse
import collections
import os
import random
import json
from copy import deepcopy

import ConfigSpace
import numpy as np
from lib.spec import Spec
from lib.model import NModel
from nas.nasbase import NasBase
from nas.nasbase import Model
from datetime import datetime


def random_architecture():
    cs = Spec.get_configuration_space()
    config = cs.sample_configuration()
    return config


def mutate_arch(parent_arch):
    # pick random parameter
    cs = Spec.get_configuration_space()
    dim = np.random.randint(len(cs.get_hyperparameters()))
    hyper = cs.get_hyperparameters()[dim]

    if type(hyper) == ConfigSpace.OrdinalHyperparameter:
        choices = list(hyper.sequence)
    else:
        choices = list(hyper.choices)
    # drop current values from potential choices
    choices.remove(parent_arch[hyper.name])

    # flip parameter
    idx = np.random.randint(len(choices))

    child_arch = deepcopy(parent_arch)
    child_arch[hyper.name] = choices[idx]
    return child_arch


def regularized_evolution(cycles, population_size, sample_size, output_path):
    """Algorithm for regularized evolution (i.e. aging evolution).

    Follows "Algorithm 1" in Real et al. "Regularized Evolution for Image
    Classifier Architecture Search".

    Args:
      cycles: the number of cycles the algorithm should run for.
      population_size: the number of individuals to keep in the population.
      sample_size: the number of individuals that should participate in each
          tournament.

    Returns:
      history: a list of `Model` instances, representing all the models computed
          during the evolution experiment.
    """
    population = collections.deque()
    c = 0
    nas = NasBase()

    # Initialize the population with random models.
    while len(population) < population_size:
        model = Model()
        model.arch = random_architecture()
        nas.train_and_eval(model)
        population.append(model)

        if len(population) % 5 == 0:
            nas.save_state(output_path, c)

    # Carry out evolution in cycles. Each cycle produces a model and removes
    # another.
    while len(history) < cycles:
        # Sample randomly chosen models from the current population.
        sample = []
        while len(sample) < sample_size:
            # Inefficient, but written this way for clarity. In the case of neural
            # nets, the efficiency of this line is irrelevant because training neural
            # nets is the rate-determining step.
            candidate = random.choice(list(population))
            sample.append(candidate)

        # The parent is the best model in the sample.
        parent = max(sample, key=lambda i: i.accuracy)

        # Create the child model and store it.
        child = Model()
        child.arch = mutate_arch(parent.arch)
        nas.train_and_eval(child)
        population.append(child)

        # Remove the oldest model.
        population.popleft()

        c += 1
        nas.save_state(output_path, c)

    return history


parser = argparse.ArgumentParser()
parser.add_argument('--run_id', default="", type=int, nargs='?', help='unique number to identify this run')
parser.add_argument('--n_iters', default=105, type=int, nargs='?', help='number of iterations for optimization method')
parser.add_argument('--output_path', default="./out", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--pop_size', default=100, type=int, nargs='?', help='population size')
parser.add_argument('--sample_size', default=10, type=int, nargs='?', help='sample_size')


args = parser.parse_args()

output_path = os.path.join(args.output_path, "regularized_evolution")
if len(args.run_id) == 0:
    now = datetime.now()
    date_time = now.strftime("%m%d%Y%H%M%S")
    output_path = os.path.join(output_path, date_time)
else:
    output_path = os.path.join(output_path, str(args.run_id))

os.makedirs(os.path.join(output_path), exist_ok=True)

history = regularized_evolution(
    cycles=args.n_iters, population_size=args.pop_size, sample_size=args.sample_size,
    output_path=output_path)