from itertools import product
from os import system, makedirs
from os.path import abspath, dirname, exists, isdir
from sys import argv
from glob import glob
from utilities import load_properties, cluster_cmd, get_num_cores
from sklearn.externals.joblib import Parallel, delayed

def RL_ens(parameters):
    code_dir, project_path, size, seed, RULE, strategy, exit, algo, age, conv, start = parameters
    cmd = 'python %s/RL_ens.py %s %s %s %s %s %s %s %s %s %s %s' % (code_dir, project_path, size, age, seed, epsilon, RULE, strategy, start, exit, conv, algo)
    if use_cluster:
        cmd = 'python %s \"%s\"' % (cluster_cmd(), cmd)
    print cmd
    system(cmd)


print "\nStarting . . ."

# ensure project directory exists
project_path = abspath(argv[1])
code_dir     = dirname(abspath(argv[0]))
directory    = 'RL_RESULTS'
subdirectory = 'ORDER'
assert exists(project_path)
dirnames     = sorted(filter(isdir, glob('%s/weka.classifiers.*' % project_path)))

# load and parse project properties
p            = load_properties(project_path)
seeds        = int(p['seeds'])
metric       = p['metric']
RULE         = p['RULE']
use_cluster  = True if p['useCluster'] in ['Y', 'y', 'yes', 'true', 'True'] else False
strategies   = ['greedy', 'pessimistic', 'backtrack']
conv_iters   = p['convIters']
age          = p['age']
epsilon      = p['epsilon']
exits        = [0]
algos        = ['Q']
start_states = '0' #start randomly ('best' also an option, see rl/run.py)
max_num_clsf = len(dirnames) * seeds
sizes        = range(1,max_num_clsf+1)


if not exists('%s/%s/' % (project_path, directory)):
    makedirs('%s/%s/' % (project_path, directory))

for o in range(seeds):
    if not exists("%s/%s/%s%i" % (project_path, directory, subdirectory, o)):
        makedirs("%s/%s/%s%i" % (project_path, directory, subdirectory, o))

all_parameters = list(product([code_dir], [project_path], sizes, range(seeds), [RULE], strategies, exits, algos, [age], [conv_iters], start_states))
Parallel(n_jobs = get_num_cores(), verbose = 50)(delayed(RL_ens)(parameters) for parameters in all_parameters)

print "\nDone!"



