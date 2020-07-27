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

def g_c_selection(best_grp):
    n = len(best_grp)
    ind = np.random.randint(n)
    return best_grp[ind]

def get_model_name(isl, gen, n):
    return 'i%d-g%d-n%d' % (isl, gen, n)


def init(population, nas, output_path, population_size, isl, gen):
    cnt = 0
    while len(population) < population_size:
        cnt += 1
        print('init:  i-%d g-%d n-%d' % (isl, gen, cnt))

        model = Model()
        model.name = get_model_name(isl, gen, cnt)
        model.name_his = ""
        model.arch = random_architecture()
        nas.train_and_eval(model, dry_run=dryRun)
        population.append(model)
        print('Model name: %s' % model.name)
        print('Model name_his: %s' % model.name_his)
        
    nas.save_state(output_path, gen, population)

def g_c_eval(population, nas, output_path, population_size, isl, gen):
    # validate
    if len(population) != 10:
        print("pop size ne 10")
        exit()

    print("zzz eval: i-%d g-%d" %(isl, gen))
    ranking_population = sorted(population)
    mid = len(ranking_population)//2
    worst_grp = ranking_population[:mid] #first half
    best_grp = ranking_population[mid:] #second half

    # validate
    if(max(best_grp) < max(worst_grp)):
        print("best lt worst")
        exit()

    offsprings = []

    cnt = 0
    for ind in best_grp:
        cnt += 1
        print('mutation:  i-%d g-%d n-%d' % (isl, gen, cnt))
        # mutation
        child_arch = mutate_arch(ind.arch)
        child = Model()

        child.name = get_model_name(isl, gen, cnt)
        child.name_his = '%s:%s' % (ind.name_his, ind.name)

        child.arch = child_arch
        nas.train_and_eval(child, dry_run=dryRun)
        offsprings.append(child)
        print('Model name: %s' % child.name)
        print('Model name_his: %s' % child.name_his)

    for ind in worst_grp:
        cnt += 1
        print('crossover:  i-%d g-%d n-%d' % (isl, gen, cnt))
        # crossover
        parent = g_c_selection(best_grp)
        child_arch = g_crossover(ind.arch, parent.arch)

        child = Model()

        child.name = get_model_name(isl, gen, cnt)
        child.name_his = '%s:%s-x-%s' % (ind.name_his, ind.name, parent.name)

        child.arch = child_arch
        nas.train_and_eval(child, dry_run=dryRun)
        offsprings.append(child)
        print('Model name: %s' % child.name)
        print('Model name_his: %s' % child.name_his)

    # evolve
    print('evolve.....')
    print('offsprings size %d' % (len(offsprings)))
    nas.save_state(output_path, gen, offsprings)

    # validate
    if len(offsprings) != 10:
        print("new pop size ne 10")
        exit()
    return offsprings


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

def ga_c(cycles, population_size, output_path, from_path):
    if len(from_path) > 0:
        output_path = from_path

    output_path1 = os.path.join(output_path, "is1")
    output_path2 = os.path.join(output_path, "is2")
    output_path3 = os.path.join(output_path, "is3")
    output_path4 = os.path.join(output_path, "is4")
    os.makedirs(os.path.join(output_path1), exist_ok=True)
    os.makedirs(os.path.join(output_path2), exist_ok=True)
    os.makedirs(os.path.join(output_path3), exist_ok=True)
    os.makedirs(os.path.join(output_path4), exist_ok=True)

    population1 = list()
    population2 = list()
    population3 = list()
    population4 = list()

    g = 1
    nas1 = NasBase()
    nas2 = NasBase()
    nas3 = NasBase()
    nas4 = NasBase()

    if len(from_path) > 0:
        print('reload history')
        population1, g = load(nas1, output_path1, population_size, 1)
        population2, g = load(nas2, output_path2, population_size, 2)
        population3, g = load(nas3, output_path3, population_size, 3)
        population4, g = load(nas4, output_path4, population_size, 4)
    else:
        print('initial history')
        init(population1, nas1, output_path1, population_size, 1, g)
        init(population2, nas2, output_path2, population_size, 2, g)
        init(population3, nas3, output_path3, population_size, 3, g)
        init(population4, nas4, output_path4, population_size, 4, g)

    print('zzz complete generation %d' % g)

    g -= 1

    while len(nas1.history) < cycles or len(nas2.history) < cycles or len(nas3.history) < cycles or len(nas4.history) < cycles :
    
        g += 1
        # crossover
        if int(len(nas1.history) / population_size) < g:
            population1 = g_c_eval(population1, nas1, output_path1, population_size, 1, g)
        if int(len(nas2.history) / population_size) < g:
            population2 = g_c_eval(population2, nas2, output_path2, population_size, 2, g)
        if int(len(nas3.history) / population_size) < g:
            population3 = g_c_eval(population3, nas3, output_path3, population_size, 3, g)
        if int(len(nas4.history) / population_size) < g:
            population4 = g_c_eval(population4, nas4, output_path4, population_size, 4, g)
        
        
        print('zzz complete generation %d' % g)
        # migrate
        if g % 8 == 0:
            print('zzz migrate all')
            best = []
            best.append(population1[len(population1) - 1])
            best.append(population2[len(population2) - 1])
            best.append(population3[len(population3) - 1])
            best.append(population4[len(population4) - 1])

            for i in range(4):
                population1.pop(int(random.random() * (len(population1) - 1)))
                population2.pop(int(random.random() * (len(population2) - 1)))
                population3.pop(int(random.random() * (len(population3) - 1)))
                population4.pop(int(random.random() * (len(population4) - 1)))

            for i in range(len(best)):
                population1.append(best[i])
                population2.append(best[i])
                population3.append(best[i])
                population4.append(best[i])


parser = argparse.ArgumentParser()
parser.add_argument('--run_id', default="", type=str, nargs='?', help='unique id to identify this run')
parser.add_argument('--n_iters', default=200, type=int, nargs='?', help='number of iterations for optimization method')
parser.add_argument('--output_path', default="./out", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--pop_size', default=10, type=int, nargs='?', help='population size')


args = parser.parse_args()

output_path = os.path.join(args.output_path, "ga-c4")
if len(args.run_id) == 0:
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_path, date_time)
else:
    output_path = os.path.join(output_path, str(args.run_id))

# os.makedirs(os.path.join(output_path), exist_ok=True)

dryRun = False

ga_c(
    cycles=args.n_iters,
    population_size=args.pop_size,
    output_path=output_path,
    from_path="")
    # from_path="./out/ga-c/20200722_081533")


# population = collections.deque()
# nas = NasBase()
# while len(population) < 10:
#         model = Model()
#         model.arch = random_architecture()
#         nas.train_and_eval(model, dry_run=dryRun)
#         population.append(model)

# print([p.accuracy for p in sorted(population)])
# print(max(population).accuracy)
