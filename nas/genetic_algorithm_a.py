import argparse
import collections
import os
import random
from copy import deepcopy
import operator

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

def mutate_arch(offspring_arch):
    # pick random parameter
    cs = Spec.get_configuration_space()
    dim = np.random.randint(len(cs.get_hyperparameters()))
    hyper = cs.get_hyperparameters()[dim]

    choices = list(hyper.choices)

    # drop current values from potential choices
    choices.remove(offspring_arch[hyper.name])

    # flip parameter
    idx = np.random.randint(len(choices))

    child_arch = deepcopy(offspring_arch)
    child_arch[hyper.name] = choices[idx]
    return child_arch

def crossover(parent_a_arch, parent_b_arch):
    cs = Spec.get_configuration_space()
    hyper = cs.get_hyperparameters()
    hyper_edge = list(filter(lambda h: (h.name.startswith('edge')), hyper))
    cnt_edge = len(hyper_edge)
    hyper_op = list(filter(lambda h: (h.name.startswith('op')), hyper))
    cnt_op = len(hyper_op)

    child_arch = deepcopy(parent_a_arch)

    # i = int(cnt_edge / 2) + 1
    i = random.randint(0, cnt_edge)
    for h in range(i, cnt_edge):
        hyper_name = 'edge_%d' % (h)
        child_arch[hyper_name] = parent_b_arch[hyper_name]

    # i = int(cnt_op / 2) + 1
    i = random.randint(0, cnt_op)
    for h in range(i, cnt_op):
        hyper_name = 'op_node_%d' % (h)
        child_arch[hyper_name] = parent_b_arch[hyper_name]

    return child_arch

def selection(population, sample_size):
    pop = sorted(population)
    n = len(pop)
    r = list((map(float, range(1,n+1))))
    s = float(sum(r))
    prop = [(p/s) for p in r]
    ind = np.random.choice(range(n), sample_size, replace=False, p=prop)
    sample = list(pop[i] for i in ind)

    ind = np.random.choice(range(sample), 2, replace=False)
    return list(sample[i] for i in ind)

def genetic_algorithm_a(cycles, population_size, sample_size, crossover_rate, mutation_rate, output_path):
    population = collections.deque()
    offsprings = collections.deque()
    g = 0
    nas = NasBase()

    # initialization
    while len(population) < population_size:
        model = Model()
        model.arch = random_architecture()
        nas.train_and_eval(model)
        population.append(model)

        if len(population) % 5 == 0:
            print('generation %d' % g)
            nas.save_state(output_path, g)
    
    while len(nas.history) < cycles:
        g += 1

        # crossover
        while len(offsprings) < population_size:
            if random.random() < float(crossover_rate):
                parents = selection(population)
                child_arch = crossover(parents[0].arch, parents[1].arch)

                # mutation
                if random.random() < float(mutation_rate):
                    child_arch = mutate_arch(child_arch)

                child = Model()
                child.arch = child_arch
                nas.train_and_eval(child)
                offsprings.append(child)

                print('generation %d' % g)
                nas.save_state(output_path, g)
            else:
                break
        
        # evolve
        pop_new = sorted(population)
        pop_new = pop_new[len(offsprings):]
        for o in offsprings:
            pop_new.append(o)
    
    return nas.history



parser = argparse.ArgumentParser()
parser.add_argument('--run_id', default="", type=str, nargs='?', help='unique id to identify this run')
parser.add_argument('--n_iters', default=100, type=int, nargs='?', help='number of iterations for optimization method')
parser.add_argument('--output_path', default="./out", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--pop_size', default=50, type=int, nargs='?', help='population size')
parser.add_argument('--sample_size', default=20, type=int, nargs='?', help='sample size')
parser.add_argument('--crossover_rate', default=0.5, type=float, nargs='?', help='crossover_rate')
parser.add_argument('--mutation_rate', default=0.5, type=float, nargs='?', help='mutation_rate')


args = parser.parse_args()

output_path = os.path.join(args.output_path, "genetic_algorithm")
if len(args.run_id) == 0:
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_path, date_time)
else:
    output_path = os.path.join(output_path, str(args.run_id))

os.makedirs(os.path.join(output_path), exist_ok=True)

history = genetic_algorithm_a(
    cycles=args.n_iters,
    population_size=args.pop_size,
    sample_size=args.sample_size,
    crossover_rate=args.crossover_rate,
    mutation_rate=args.mutation_rate,
    output_path=output_path)
