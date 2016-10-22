from glob import glob
import gzip
from os.path import abspath, exists, isdir
from sys import argv
from numpy import array, random
from sklearn.metrics import roc_auc_score
from sklearn.externals.joblib import Parallel, delayed
from os import makedirs
from utilities import load_properties, fmax_score, get_set_preds, get_num_cores, get_path_bag_weight, get_bps, aggregate_predictions
from pandas import concat, read_csv, DataFrame
from itertools import product
import math


def get_max_predictions(predictors, seed, fold, set):
    max_p = ''
    max_w = 0

    path, bag, weight = get_path_bag_weight(predictors[0])
    if weight > max_w:
        max_w = weight
        max_p = path

    for bp in predictors[1:]:
        path, bag, weight = get_path_bag_weight(bp)
        if weight > max_w:
            max_w = weight
            max_p = path
    
    set = 'test'
    #print 'GET_MAX_PREDICTIONS FOR THE BEST BP, I.E., %s_bag%s (based on the order file obtained form the validation set)\n' % (max_p, bag)
    y_true, y_score = get_set_preds(max_p, set, bag, fold, seed)
    perf = fmax_score(y_true, y_score)
    return (y_true, y_score, ('%s_bag%s' % (max_p, max_w)))


def BEST_bp(parameters):
    size, seed = parameters   
    
    y_true = DataFrame(columns = ["label"])
    y_score = DataFrame(columns = ["prediction"])
    string = ""
    for fold in range(fold_count):
        ensemble_bps = get_bps(project_path, seed, metric, size)[0]
        inner_y_true, inner_y_score, bp = get_max_predictions(ensemble_bps, seed, fold, "test")
        y_true = concat([y_true, inner_y_true], axis = 0)
        y_score = concat([y_score, inner_y_score], axis = 0)
        string += ("fold_%i,%f\n" % (fold, fmax_score(inner_y_true, inner_y_score)))
    string += ("final,%f\n" % fmax_score(y_true, y_score))
    filename = '%s/%s/%s%i/BP_bp%i_seed%i.fmax' % (project_path, directory, subdirectory, seed, size, seed)

    with open(filename, 'wb') as f:
    	f.write(string)
    f.close()
    print filename 
    

def FULL_ens(parameters):
    size, seed = parameters

    y_true = DataFrame(columns = ["label"])
    y_score = DataFrame(columns = ["prediction"])
    string = ""
    for fold in range(fold_count):
        ensemble_bps = get_bps(project_path, seed, metric, size)[0]
        inner_y_true, inner_y_score = aggregate_predictions(ensemble_bps, seed, fold, "test", RULE)
        y_true = concat([y_true, inner_y_true], axis = 0)
        y_score = concat([y_score, inner_y_score], axis = 0)
        string += ("fold_%i,%f\n" % (fold, fmax_score(inner_y_true, inner_y_score)))
    string += ("final,%f\n" % fmax_score(y_true, y_score))
    filename = '%s/%s/%s%i/FE_bp%i_seed%i_%s.fmax' % (project_path, directory, subdirectory, seed, size, seed, RULE)

    with open(filename, 'wb') as f:
        f.write(string)
    f.close()
    print filename

def baselines(parameters):
    BEST_bp(parameters)
    FULL_ens(parameters)


print "\nStarting . . ."

# ensure project directory exists
project_path = abspath(argv[1])
assert exists(project_path)
directory    = 'BASELINE'
subdirectory = 'ORDER'
dirnames     = sorted(filter(isdir, glob('%s/weka.classifiers.*' % project_path)))

# load and parse project properties
p          = load_properties(project_path)
fold_count = int(p['foldCount'])
seeds      = int(p['seeds'])
metric     = p['metric']
RULE       = p['RULE']


max_num_clsf = len(dirnames) * seeds
sizes        = range(1,max_num_clsf+1)

if not exists('%s/%s/' % (project_path, directory)):
    makedirs('%s/%s/' % (project_path, directory))

for o in range(seeds):
    if not exists("%s/%s/%s%i" % (project_path, directory, subdirectory, o)):
        makedirs("%s/%s/%s%i" % (project_path, directory, subdirectory, o))

all_parameters = list(product(sizes, range(seeds)))
Parallel(n_jobs = get_num_cores(), verbose = 50)(delayed(baselines)(parameters) for parameters in all_parameters)

print "Done!\n"




