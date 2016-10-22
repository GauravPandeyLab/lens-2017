from glob import glob
import gzip
from os.path import abspath, exists, isdir, getsize
from sys import argv, exit
from random import randrange
from numpy import array
from sklearn.metrics import roc_auc_score
from os import makedirs
from utilities import load_properties, fmax_score, get_fold_ens
from pandas import concat, read_csv, DataFrame
from itertools import product


def get_ens_perf(filename):
    if not exists(filename) or (getsize(filename) == 0):
         return float('-inf')
    else:
        with open(filename, 'r') as f:
            content = f.read().splitlines()
        f.close()
        if (content[len(content)-1].split(",")[0] == "final"):
            return float(content[len(content)-1].split(",")[1])
        else:
            return float('-inf')


def resultsRL_values(strategy, exit, RULE, algo, conv, age, epsilon, start):
    filename = '%s/RESULTS/RL/RESULTS_RL_epsilon%s_pre%s_conv%s_exit%s_%s_%s_%s_start-%s_%s.csv' % (project_path, epsilon, age, conv, exit, strategy, RULE, algo, start, metric)
    with open(filename, 'wb') as f:
        title = "".join(("ens%i, " %i) for i in sizes).rstrip(", ")
        f.write("Seed/Order,%s\n" % title)

        for seed in range(seeds):
            seed_line = ""
            for size in sizes:
                rl_file = '%s/RL_RESULTS/ORDER%s/RL_bp%i_seed%s_epsilon%s_pre%s_conv%s_exit%s_%s_%s_%s_start-%s.%s' % (project_path, seed, size, seed, epsilon, age, conv, exit, strategy, RULE, algo, start, metric)
                val = get_ens_perf(rl_file)
                seed_line+=("%s," % val)
            f.write("SEED_%i,%s\n" % (seed, seed_line[:-1]))
    f.close()
    print('RL VALUES\t\t :: %s' % filename)

def resultsRL_dims(strategy, exit, RULE, algo, conv, age, epsilon, start):
    filename = '%s/RESULTS/RL/RESULTS_RL_epsilon%s_pre%s_conv%s_exit%s_%s_%s_%s_start-%s_%s_dim.csv' % (project_path, epsilon, age, conv, exit, strategy, RULE, algo, start, metric)

    with open(filename, 'wb') as f:
        title = "".join(("ens%i, " %i) for i in sizes).rstrip(", ")
        f.write("Seed/Order,%s\n" % title)

        for seed in range(seeds):
            seed_line = ""
            for size in sizes:
                dimension = 0.0
                for fold in range(fold_count):
                    rl_file = '%s/RL_OUTPUT/ORDER%s/bp%i_fold%s_seed%s_epsilon%s_pre%s_conv%s_exit%s_%s_%s_%s_start-%s.%s' % (project_path, seed, size, fold, seed, epsilon, age, conv, exit, strategy, RULE, algo, start, metric)
                    current = len(get_fold_ens(rl_file))
                    dimension += current
                dim = dimension / float(fold_count)
                seed_line+=("%s," % dim)
            f.write("SEED_%i,%s\n" % (seed, seed_line[:-1]))
    f.close()
    print('CES DIMENSIONS\t\t :: %s' % filename)

def resultsRL():
    if not exists("%s/%s/RL/" % (project_path, directory)):
        makedirs("%s/%s/RL" % (project_path, directory))
    for parameters in all_parameters:
        strategy, exit, algo, conv, age, epsilon, start = parameters
        resultsRL_values(strategy, exit, RULE, algo, conv, age, epsilon, start)
        resultsRL_dims(strategy, exit, RULE, algo, conv, age, epsilon, start)



# # #
def resultsFE_values():
    filename = '%s/RESULTS/FE/RESULTS_FE_%s_%s.csv' % (project_path, RULE, metric)
    with open(filename, 'wb') as f:
        title = "".join(("ens%i, " %i) for i in sizes).rstrip(", ")
        f.write("Seed/Order,%s\n" % title)

        for seed in range(seeds):
            seed_line = ""
            for size in sizes:
                fe_file = '%s/BASELINE/ORDER%s/FE_bp%i_seed%s_%s.%s' % (project_path, seed, size, seed, RULE, metric)
                val = get_ens_perf(fe_file)
                seed_line+=("%s," % val)
            f.write("SEED_%i,%s\n" % (seed, seed_line[:-1]))
    f.close()
    print('FE VALUES\t\t :: %s' % filename)

def resultsFE():
    if not exists("%s/%s/FE/" % (project_path, directory)):
        makedirs("%s/%s/FE" % (project_path, directory))
    resultsFE_values()


# # #

def resultsCES_values():
    filename = '%s/RESULTS/CES/RESULTS_CES_%s_start-%s_%s.csv' % (project_path, RULE, CES_start, metric)
    with open(filename, 'wb') as f:
        title = "".join(("ens%i, " %i) for i in sizes).rstrip(", ")
        f.write("Seed/Order,%s\n" % title)

        for seed in range(seeds):
            seed_line = ""
            for size in sizes:
                ces_file = '%s/CES_RESULTS/ORDER%s/CES_bp%i_seed%s_%s_start-%s.%s' % (project_path, seed, size, seed, RULE, CES_start, metric)
                val = get_ens_perf(ces_file)
                seed_line+=("%s," % val)
            f.write("SEED_%i,%s\n" % (seed, seed_line[:-1]))
    f.close()
    print('CES VALUES\t\t :: %s' % filename)


def resultsCES_dims():
    filename = '%s/RESULTS/CES/RESULTS_CES_%s_start-%s_%s_dim.csv' % (project_path, RULE, CES_start, metric)    
    with open(filename, 'wb') as f:
        title = "".join(("ens%i, " %i) for i in sizes).rstrip(", ")
        f.write("Seed/Order,%s\n" % title)

        for seed in range(seeds):
            seed_line = ""
            for size in sizes:
                dimension = 0.0
                for fold in range(fold_count):
                    ces_file = '%s/CES_OUTPUT/ORDER%s/bp%i_fold%i_seed%s_%s_start-%s.%s' % (project_path, seed, size, fold, seed, RULE, CES_start, metric)
                    current = len(get_fold_ens(ces_file)) 
                    dimension += current
                dim = dimension / float(fold_count)
                seed_line+=("%s," % dim)
            f.write("SEED_%i,%s\n" % (seed, seed_line[:-1]))
    f.close()
    print('CES DIMENSIONS\t\t :: %s' % filename)


def resultsCES():
    if not exists("%s/%s/CES/" % (project_path, directory)):
        makedirs("%s/%s/CES" % (project_path, directory))
    resultsCES_values()
    resultsCES_dims()

# # #

def resultsBP():
    if not exists('%s/%s/BP/' % (project_path, directory)):
        makedirs('%s/%s/BP' % (project_path, directory))


    filename = '%s/RESULTS/BP/RESULTS_BP_fmax.csv' % (project_path)
    with open(filename, 'wb') as f:
        title = "".join(("ens%i, " %i) for i in sizes).rstrip(", ")
        f.write("Seed/Order,%s\n" % title)

        for seed in range(seeds):
            seed_line = ""
            for size in sizes:
                bp_file = '%s/BASELINE/ORDER%s/BP_bp%i_seed%s.fmax' % (project_path, seed, size, seed)
                val = get_ens_perf(bp_file)
                seed_line+=("%s," % val)
            f.write("SEED_%i,%s\n" % (seed, seed_line[:-1]))
    f.close()
    print('BEST PREDICTOR VALUES\t :: %s' % (filename))

# # #


print "\nStarting. . ."

# ensure project directory exists
project_path = abspath(argv[1])
assert exists(project_path)
directory    = 'RESULTS'
subdirectory = 'ORDER'
dirnames     = sorted(filter(isdir, glob('%s/weka.classifiers.*' % project_path)))


# load and parse project properties
p            = load_properties(project_path)
fold_count   = int(p['foldCount'])
seeds        = int(p['seeds'])
metric       = p['metric']
RULE         = p['RULE']
CES_start  = '1'  #initialize ensemble with top model
max_num_clsf = len(dirnames) * seeds
sizes        = range(1,max_num_clsf+1)


#rl
strategies   = ['greedy', 'pessimistic', 'backtrack']
conv_iters   = int(p['convIters'])
age          = int(p['age'])
epsilon      = p['epsilon']
exits        = [0]
algos        = ['Q']
start_states = '0' #start randomly ('best' also an option, see rl/run.py)
all_parameters = list(product(strategies, exits, algos, [conv_iters], [age], [epsilon], [start_states]))


if not exists('%s/%s/' % (project_path, directory)):
    makedirs('%s/%s/' % (project_path, directory))

resultsBP()
resultsFE()
resultsCES()
resultsRL()

print "Done!\n"









