from itertools import product
from os import environ, system, makedirs
from os.path import abspath, dirname, exists, isdir
from sys import argv
from glob import glob
from utilities import load_properties, cluster_cmd, get_num_cores
from sklearn.externals.joblib import Parallel, delayed

def RL_run(parameters):
    code_dir, project_path, size, fold, seed, RULE, strategy, exit, algo, conv, age, epsilon, start = parameters
    cmd = 'python %s/rl/run.py -i %s -o %s/RL_OUTPUT/ORDER%s/ -np %s -fold %s -m %s -seed %i -epsilon %s -rule %s -strategy %s -exit %i -algo %s -age %i -conv %i -start %s' % (code_dir, project_path, project_path, seed, size, fold, metric, seed, epsilon, RULE, strategy, exit, algo, age, conv, start) 
    if use_cluster:
        cmd = 'python %s \"%s\"' % (cluster_cmd(), cmd)
    print cmd
    system(cmd)


print "Starting . . .\n"

# ensure project directory exists
project_path = abspath(argv[1])
assert exists(project_path)
code_dir     = dirname(abspath(argv[0]))
directory    = 'RL_OUTPUT'
subdirectory = 'ORDER'
dirnames     = sorted(filter(isdir, glob('%s/weka.classifiers.*' % project_path)))


# load and parse project properties
p            = load_properties(project_path)
fold_count   = int(p['foldCount'])
seeds 	     = int(p['seeds'])
metric       = p['metric']
RULE 	     = p['RULE']
use_cluster  = True if p['useCluster'] in ['Y', 'y', 'yes', 'true', 'True'] else False
strategies   = ['greedy', 'pessimistic', 'backtrack']
conv_iters   = int(p['convIters'])
age          = int(p['age'])
epsilon      = p['epsilon']
exits        = [0]
algos        = ['Q']
start_states = '0' #start randomly ('best' also an option, see rl/run.py)
max_num_clsf = len(dirnames) * seeds
sizes        = range(1,max_num_clsf+1)


if not exists("%s/RL_OUTPUT/" % project_path):
    makedirs("%s/RL_OUTPUT/" % project_path)

for o in range(seeds):
    if not exists("%s/RL_OUTPUT/ORDER%i" % (project_path, o)):
        makedirs("%s/RL_OUTPUT/ORDER%i" % (project_path, o))

all_parameters = list(product([code_dir], [project_path], sizes, range(fold_count), range(seeds), [RULE], strategies, exits, algos, [conv_iters], [age], [epsilon], [start_states]))
Parallel(n_jobs = get_num_cores(), verbose = 50)(delayed(RL_run)(parameters) for parameters in all_parameters)

print "\nDone!"



