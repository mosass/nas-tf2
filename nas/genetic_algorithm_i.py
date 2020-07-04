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

def selection(population):
    sort = sorted(population)
    pop = list(sort[i] for i in range(len(sort) - 4, len(sort)))
    pop.append(population[0])
    pop = sorted(pop)
    n = len(pop)
    r = list((map(float, range(1,n+1))))
    s = float(sum(r))
    prop = [(p/s) for p in r]
    # ind = np.random.choice(range(n), 2, replace=False, p=prop)
    ind = np.random.choice(range(n), 2, replace=False)
    return list(pop[i] for i in ind)

def genetic_algorithm(cycles, population_size, crossover_rate, mutation_rate, output_path):
    output_path1 = os.path.join(output_path, "is1")
    output_path2 = os.path.join(output_path, "is2")
    os.makedirs(os.path.join(output_path1), exist_ok=True)
    os.makedirs(os.path.join(output_path2), exist_ok=True)

    population1 = collections.deque()
    population2 = collections.deque()
    offsprings1 = collections.deque()
    offsprings2 = collections.deque()
    g = 1
    nas1 = NasBase()
    nas2 = NasBase()

    # initialization 1
    while len(population1) < population_size:
        model = Model()
        model.arch = random_architecture()
        nas1.train_and_eval(model, dry_run=dryRun)
        population1.append(model)
        print('history1 size %d' % len(nas1.history))

        if len(population1) % 5 == 0:
            print('generation %d' % g)
            nas1.save_state(output_path1, g)

    # initialization 2
    while len(population2) < population_size:
        model = Model()
        model.arch = random_architecture()
        nas2.train_and_eval(model, dry_run=dryRun)
        population2.append(model)
        print('history2 size %d' % len(nas2.history))

        if len(population2) % 5 == 0:
            print('generation %d' % g)
            nas2.save_state(output_path2, g)
    
    while len(nas1.history) < cycles and len(nas2.history) < cycles:
    
        g += 1
        # crossover
        while len(offsprings1) < population_size:
            if random.random() < float(crossover_rate):
                parents = selection(population1)
                child_arch = crossover(parents[0].arch, parents[1].arch)

                # mutation
                if random.random() < float(mutation_rate):
                    child_arch = mutate_arch(child_arch)

                child = Model()
                child.arch = child_arch
                nas1.train_and_eval(child, dry_run=dryRun)
                offsprings1.append(child)

                print('history1 size %d' % len(nas1.history))
                print('generation %d' % g)
                nas1.save_state(output_path1, g)
            else:
                break
        
        # evolve
        print('evolve.....')
        print('offsprings1 size %d' % len(offsprings1))
        pop_new = sorted(population1)
        pop_new = pop_new[len(offsprings1):]
        for o in offsprings1:
            pop_new.append(o)

        population1 = sorted(pop_new)
        offsprings1 = collections.deque()
        

        # crossover
        while len(offsprings2) < population_size:
            if random.random() < float(crossover_rate):
                parents = selection(population2)
                child_arch = crossover(parents[0].arch, parents[1].arch)

                # mutation
                if random.random() < float(mutation_rate):
                    child_arch = mutate_arch(child_arch)

                child = Model()
                child.arch = child_arch
                nas2.train_and_eval(child, dry_run=dryRun)
                offsprings2.append(child)

                print('history2 size %d' % len(nas2.history))
                print('generation %d' % g)
                nas2.save_state(output_path2, g)
            else:
                break
        
        # evolve
        print('evolve.....')
        print('offsprings2 size %d' % len(offsprings2))
        pop_new = sorted(population2)
        pop_new = pop_new[len(offsprings2):]
        for o in offsprings2:
            pop_new.append(o)

        population2 = sorted(pop_new)
        offsprings2 = collections.deque()

        # migrate
        if g % 4 == 0:
            best1 = population1[len(population1) - 1]
            best2 = population2[len(population1) - 1]

            population1.pop(int(random.random() * (len(population1) - 1)))
            population2.pop(int(random.random() * (len(population2) - 1)))

            population1.append(best2)
            population2.append(best1)


    return [nas1.history, nas2.history]



parser = argparse.ArgumentParser()
parser.add_argument('--run_id', default="", type=str, nargs='?', help='unique id to identify this run')
parser.add_argument('--n_iters', default=100, type=int, nargs='?', help='number of iterations for optimization method')
parser.add_argument('--output_path', default="./out", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--pop_size', default=10, type=int, nargs='?', help='population size')
parser.add_argument('--crossover_rate', default=1.0, type=float, nargs='?', help='crossover_rate')
parser.add_argument('--mutation_rate', default=0.5, type=float, nargs='?', help='mutation_rate')

dryRun = True


args = parser.parse_args()

output_path = os.path.join(args.output_path, "genetic_algorithm_i")
if len(args.run_id) == 0:
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_path, date_time)
else:
    output_path = os.path.join(output_path, str(args.run_id))

os.makedirs(os.path.join(output_path), exist_ok=True)

history = genetic_algorithm(
    cycles=args.n_iters,
    population_size=args.pop_size,
    crossover_rate=args.crossover_rate,
    mutation_rate=args.mutation_rate,
    output_path=output_path)
