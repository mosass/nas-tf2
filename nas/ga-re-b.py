import argparse
import collections
import os
import random
from copy import deepcopy
import operator
import glob
import json

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

def g_crossover(parent_a_arch, parent_b_arch):
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

def g_selection(population):
    sort = sorted(population) # asc
    pop = list(sort[i] for i in range(len(sort) - 4, len(sort)))
    pop.append(population[0])
    pop = sorted(pop)
    n = len(pop)
    # r = list((map(float, range(1,n+1))))
    # s = float(sum(r))
    # prop = [(p/s) for p in r]
    # ind = np.random.choice(range(n), 2, replace=False, p=prop)
    ind = np.random.choice(range(n), 2, replace=False)
    return list(pop[i] for i in ind)

def init(population, nas, output_path, population_size, isl, gen):
    while len(population) < population_size:
        model = Model()
        model.arch = random_architecture()
        nas.train_and_eval(model, dry_run=dryRun)
        population.append(model)
        print('history-%d size %d' % (isl, len(nas.history)))
        print('generation %d' % gen)

    nas.save_state(output_path, gen, population)

def g_eval(population, nas, output_path, population_size, crossover_rate, mutation_rate, isl, gen):
    offsprings = collections.deque()
    while len(offsprings) < population_size:
        if random.random() < float(crossover_rate):
            parents = g_selection(population)
            child_arch = g_crossover(parents[0].arch, parents[1].arch)

            # mutation
            if random.random() < float(mutation_rate):
                child_arch = mutate_arch(child_arch)

            child = Model()
            child.arch = child_arch
            nas.train_and_eval(child, dry_run=dryRun)
            offsprings.append(child)

            print('history-%d size %d' % (isl, len(nas.history)))
            print('generation %d' % gen)
        else:
            if len(offsprings) > 0:
                break
    
    # evolve
    print('evolve.....')
    print('offsprings-%d size %d' % (gen, len(offsprings)))
    pop_new = sorted(population)
    pop_new = pop_new[len(offsprings):]
    for o in offsprings:
        pop_new.append(o)

    nas.save_state(output_path, gen, sorted(pop_new))
    
    return sorted(pop_new)

def r_selection(population, sample_size, select_size):
    n = len(population)
    ind = np.random.choice(range(n), sample_size, replace=False)
    pop = list(population[i] for i in ind)
    pop = sorted(pop)
    return pop[-select_size:]

def r_eval(population, nas, output_path, population_size, isl, gen):
    sample = r_selection(population, int(population_size/2), 2)

    for s in sample:
        # mutation
        child_arch = mutate_arch(s.arch)
        child = Model()
        child.arch = child_arch

        nas.train_and_eval(child, dry_run=dryRun)

        print('history-%d size %d' % (isl, len(nas.history)))
        print('generation %d' % gen)
        population.append(child)
        population.pop(0)

    # evolve
    print('evolve.....')
    nas.save_state(output_path, gen, population)
    
    return population

def load(nas, output_path, population_size, isl):

    histfp = open(output_path+'/history.json', 'r')
    hist = json.load(histfp)

    pop_histfp = open(output_path+'/state_history.json', 'r')
    pop_hist = json.load(pop_histfp)

    for i in range(len(hist)):
        model = Model()
        model.accuracy = hist[i]['accuracy']
        model.data = hist[i]['data']

        cs = Spec.get_configuration_space()
        arch = cs.sample_configuration()

        for k,v in hist[i]['arch'].items():
            arch[k] = v

        model.arch = arch
        
        nas.history.append(model)
    
    for i in range(len(pop_hist)):
        model = Model()
        model.accuracy = pop_hist[i]['accuracy']
        model.data = pop_hist[i]['data']

        cs = Spec.get_configuration_space()
        arch = cs.sample_configuration()

        for k,v in pop_hist[i]['arch'].items():
            arch[k] = v

        model.arch = arch
        
        nas.pop_history.append(model)

    
    population = nas.pop_history[-population_size:]

    return population, int(len(nas.pop_history) / population_size)

def ga_re_a(cycles, population_size, crossover_rate, mutation_rate, output_path, from_path):
    if len(from_path) > 0:
        output_path = from_path

    output_path1 = os.path.join(output_path, "is1")
    output_path2 = os.path.join(output_path, "is2")
    output_path3 = os.path.join(output_path, "re3")
    output_path4 = os.path.join(output_path, "is4")
    output_path5 = os.path.join(output_path, "is5")
    output_path6 = os.path.join(output_path, "re6")
    os.makedirs(os.path.join(output_path1), exist_ok=True)
    os.makedirs(os.path.join(output_path2), exist_ok=True)
    os.makedirs(os.path.join(output_path3), exist_ok=True)
    os.makedirs(os.path.join(output_path4), exist_ok=True)
    os.makedirs(os.path.join(output_path5), exist_ok=True)
    os.makedirs(os.path.join(output_path6), exist_ok=True)

    population1 = list()
    population2 = list()
    population3 = list()
    population4 = list()
    population5 = list()
    population6 = list()

    g = 1
    nas1 = NasBase()
    nas2 = NasBase()
    nas3 = NasBase()
    nas4 = NasBase()
    nas5 = NasBase()
    nas6 = NasBase()

    if len(from_path) > 0:
        print('reload history')
        population1, g = load(nas1, output_path1, population_size, 1)
        population2, g = load(nas2, output_path2, population_size, 2)
        population3, g = load(nas3, output_path3, population_size, 3)
        population4, g = load(nas4, output_path4, population_size, 4)
        population5, g = load(nas5, output_path5, population_size, 5)
        population6, g = load(nas6, output_path6, population_size, 6)
    else:
        print('initial history')
        init(population1, nas1, output_path1, population_size, 1, g)
        init(population2, nas2, output_path2, population_size, 2, g)
        init(population3, nas3, output_path3, population_size, 3, g)
        init(population4, nas4, output_path4, population_size, 4, g)
        init(population5, nas5, output_path5, population_size, 5, g)
        init(population6, nas6, output_path6, population_size, 6, g)

    print('zzz complete generation %d' % g)

    g -= 1

    while len(nas1.history) < cycles or len(nas2.history) < cycles or len(nas3.history) < cycles or len(nas4.history) < cycles:
    
        g += 1
        # crossover
        if int(len(nas1.history) / population_size) < g:
            population1 = g_eval(population1, nas1, output_path1, population_size, crossover_rate, mutation_rate, 1, g)
        if int(len(nas2.history) / population_size) < g:
            population2 = g_eval(population2, nas2, output_path2, population_size, crossover_rate, mutation_rate, 2, g)
        if int(len(nas3.history) / population_size) < g:
            population3 = r_eval(population3, nas3, output_path3, population_size, 3, g)
            
        if int(len(nas4.history) / population_size) < g:
            population4 = g_eval(population4, nas4, output_path4, population_size, crossover_rate, mutation_rate, 4, g)
        if int(len(nas5.history) / population_size) < g:
            population5 = g_eval(population5, nas5, output_path5, population_size, crossover_rate, mutation_rate, 5, g)
        if int(len(nas6.history) / population_size) < g:
            population6 = r_eval(population6, nas6, output_path6, population_size, 6, g)
        
        
        print('zzz complete generation %d' % g)
        # migrate
        if g % 3 == 0:
            print('zzz migrate ring GA')

            best1 = max(population1)
            best2 = max(population2)
            best3 = max(population3) # re
            best4 = max(population4)
            best5 = max(population5)
            best6 = max(population6) # re

            for i in range(2):
                population1.pop(int(random.random() * (len(population1) - 1)))
                population2.pop(int(random.random() * (len(population2) - 1)))
                population3.pop(int(random.random() * (len(population3) - 1)))
                population4.pop(int(random.random() * (len(population4) - 1)))
                population5.pop(int(random.random() * (len(population5) - 1)))
                population6.pop(int(random.random() * (len(population6) - 1)))

            #  1   ->  2
            #  ^       v
            #  |   3   |
            #  |       |
            #  |   6   |
            #  ^       v
            #  5   <-  4
            
            population1.append(best5)
            population2.append(best1)
            population4.append(best2)
            population5.append(best4)

            print('zzz migrate GA - RE')
            population1.append(best3) 
            population2.append(best3) 
            population3.append(best1)
            population3.append(best2)

            population4.append(best6) 
            population5.append(best6) 
            population6.append(best4)
            population6.append(best5)


parser = argparse.ArgumentParser()
parser.add_argument('--run_id', default="", type=str, nargs='?', help='unique id to identify this run')
parser.add_argument('--n_iters', default=150, type=int, nargs='?', help='number of iterations for optimization method')
parser.add_argument('--output_path', default="./out", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--pop_size', default=10, type=int, nargs='?', help='population size')
parser.add_argument('--crossover_rate', default=0.9, type=float, nargs='?', help='crossover_rate')
parser.add_argument('--mutation_rate', default=1.0, type=float, nargs='?', help='mutation_rate')

dryRun = False


args = parser.parse_args()

output_path = os.path.join(args.output_path, "ga-re-b")
if len(args.run_id) == 0:
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_path, date_time)
else:
    output_path = os.path.join(output_path, str(args.run_id))

# os.makedirs(os.path.join(output_path), exist_ok=True)

ga_re_a(
    cycles=args.n_iters,
    population_size=args.pop_size,
    crossover_rate=args.crossover_rate,
    mutation_rate=args.mutation_rate,
    output_path=output_path,
    # from_path="")
    from_path="./out/ga-re-b/20200719_174625")


# population = collections.deque()
# nas = NasBase()
# while len(population) < 10:
#         model = Model()
#         model.arch = random_architecture()
#         nas.train_and_eval(model, dry_run=dryRun)
#         population.append(model)

# print([p.accuracy for p in sorted(population)])
# print(max(population).accuracy)
