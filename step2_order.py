from os.path import abspath, exists, isdir
from os import makedirs
from sys import argv
from glob import glob
import random
import gzip
import pickle
from numpy import array
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, f1_score
from utilities import load_properties, f_score, fmax_score
from pandas import concat, read_csv, DataFrame
from collections import OrderedDict

print "\nStarting. . ."

# ensure project directory exists
project_path = abspath(argv[1])
assert exists(project_path)

if not exists("%s/ENSEMBLES/" % project_path):
    makedirs("%s/ENSEMBLES/" % project_path)

# load and parse project properties
p          	= load_properties(project_path)
fold_count 	= int(p['foldCount'])
seeds 	   	= int(p['seeds'])
bags 	   	= int(p['bags'])
metric		= p['metric']
assert (metric in ['fmax', 'auROC'])

dirnames = sorted(filter(isdir, glob('%s/weka.classifiers.*' % project_path)))

print "Decreasing order of performance of the base predictors for the seeds (seed in [0-%i]) on the validation set:" % seeds
seed_list = range(seeds)
bag_list  = range(bags)

for seed in seed_list:
    dir_dict = {}
    order_fn = '%s/ENSEMBLES/order_of_seed%i_%s.txt' % (project_path, seed, metric) 
    with open(order_fn, 'wb') as order_file:
        for dirname in dirnames:
            for bag in bag_list:
                x1 = DataFrame(columns = ["label"])
	        x2 = DataFrame(columns = ["prediction"])
	        for fold in range(fold_count):
                    filename = '%s/valid-b%i-f%s-s%i.csv.gz' % (dirname, bag, fold, seed)
                    df = read_csv(filename, skiprows = 1, compression = 'gzip')
                    y_true = df.ix[:,1:2]
                    y_score = df.ix[:,2:3]
	        x1 = concat([x1, y_true], axis = 0)
	        x2 = concat([x2, y_score], axis = 0)

                if metric == "fmax":
	            dir_dict["%s_bag%i" % (dirname, bag)] = fmax_score(x1,x2)    	
                if metric == "auROC":
                    dir_dict ["%s_bag%i" % (dirname, bag)] = roc_auc_score(x1,x2)
	        d_sorted_by_value = OrderedDict(sorted(dir_dict.items(), key=lambda x: (-x[1], x[0])))

	for key, v in d_sorted_by_value.items():
            order_file.write("%s, %s \n" % (key, v))
	order_file.close()
	print order_fn


print "Done!\n"






