from glob import glob
import gzip
from os.path import abspath, exists, isdir, getsize
from sys import argv
from random import randrange
from numpy import array, mean, std, transpose
from sklearn.metrics import roc_auc_score
from os import makedirs
from utilities import load_properties, fmax_score
from pandas import concat, read_csv, DataFrame
from itertools import product
from scipy import stats
#from scipy.integrate import simps
from numpy import trapz, array


# # # # #
	
def resultsFE():
    file_name = '%s/RESULTS/FE/RESULTS_FE_%s_%s.csv' % (project_path, RULE, metric)
    mydf = read_csv(file_name)
    mydf.fillna(0, inplace=True)
    fe_labels = list(mydf.columns.values)[1:]
    fe_mean  = list(mean(mydf, axis=0))
    y  = array(fe_mean)
    auc = trapz(y, dx=1)/float(len(y)-1)
    return fe_labels, fe_mean, auc

def resultsBEST():
    file_name = '%s/RESULTS/BP/RESULTS_BP_%s.csv' % (project_path, metric)
    mydf = read_csv(file_name)
    mydf.fillna(0, inplace=True)
    best_labels = list(mydf.columns.values)[1:]
    best_mean  = list(mean(mydf, axis=0))
    y  = array(best_mean)
    auc = trapz(y, dx=1)/float(len(y)-1)
    return best_labels, best_mean, auc

def resultsCES():
    file_name = '%s/RESULTS/CES/RESULTS_CES_%s_start-%s_%s.csv' % (project_path, RULE, CES_start, metric)
    mydf = read_csv(file_name)
    mydf.fillna(0, inplace=True)
    ces_labels = list(mydf.columns.values)[1:]
    ces_mean  = list(mean(mydf, axis=0))
    dim_file = '%s/RESULTS/CES/RESULTS_CES_%s_start-%s_%s_dim.csv' % (project_path, RULE, CES_start, metric)
    dimdf = read_csv(dim_file)
    dims = list(mean(dimdf, axis=0))
    dim = [("%.2f" % dims[step]) for step in steps]
    y  = array(ces_mean)
    auc = trapz(y, dx=1)/float(len(y)-1)
    return dim, ces_mean, auc

def resultsRL(epsilon, age, conv, exit, strategy, RULE, algo, start, metric):
    results_file = '%s/RESULTS/RL/RESULTS_RL_epsilon%s_pre%s_conv%s_exit%s_%s_%s_%s_start-%s_%s.csv' % (project_path, epsilon, age, conv, exit, strategy, RULE, algo, start, metric)
    mydf = read_csv(results_file)
    mydf.fillna(0, inplace=True)
    rl_labels = list(mydf.columns.values)[1:]
    rl_mean  = list(mean(mydf, axis=0))
    dim_file = '%s/RESULTS/RL/RESULTS_RL_epsilon%s_pre%s_conv%s_exit%s_%s_%s_%s_start-%s_%s_dim.csv' % (project_path, epsilon, age, conv, exit, strategy, RULE, algo, start, metric)
    dimdf = read_csv(dim_file)
    dims = list(mean(dimdf, axis=0))
    dim = [("%.2f" % dims[step]) for step in steps]
    y  = array(rl_mean)
    auc = trapz(y, dx=1)/float(len(y)-1)
    return dim, rl_mean, auc

# # # # #

def write_plot():
    str = ""
    
    best_labels, best_mean,  best_auc = resultsBEST()
    str += ("    avg_best = %r\n" % best_mean)
    str += ("    best = plt.plot(%r, avg_best, color='red', label='BEST (auESC=%.4f)')\n\n\n" % (x_ticks, best_auc))
    

    fe_labels, fe_mean,  fe_auc = resultsFE()
    str += ("    avg_fe = %r\n" % fe_mean)
    str += ("    fe = plt.plot(%r, avg_fe, color='magenta', label='FE (auESC=%.4f)')\n\n\n" % (x_ticks, fe_auc))

    ces_labels, ces_mean, ces_auc = resultsCES()
    str += ("    dim_ces = %r\n"
            "    avg_ces = %r\n"
            "    x=0\n"
            "    for i,j in zip(%r, %s):\n"
            "        plt.annotate(dim_ces[x], xy=(i,j), fontsize = 7, color='orange')\n"
            "        x+=1\n" % (ces_labels, ces_mean, x_ticks, ces_mean))
    str += ("    ces = plt.plot(%r, avg_ces, color='orange', label='CES (auESC=%.4f)')\n\n\n" % (x_ticks, ces_auc))

    index = 0
    for strategy in strategies:
        rl_labels, mean_rl, rl_auc = resultsRL(epsilon, age, conv_iters, exit, strategy, RULE, algo, start_state, metric)
        str += ("    dim_rl = %r\n"
                "    avg_rl = %r\n"
                "    x=0\n"
                "    for i,j in zip(%r, avg_rl):\n"
                "        plt.annotate(dim_rl[x], xy=(i,j), fontsize = 10, color = \'%s\')\n"
                "        x+=1\n" % (rl_labels, mean_rl, x_ticks, color[index]))
        str += ("    rl = plt.plot(%r, avg_rl, color=\'%s\', label=\'RL_%s_start-%s_pre%i (auESC=%.4f)\')\n\n\n" % (x_ticks, color[index], strategy, start_state, age, rl_auc))
        index += 1





    str += ("    plt.title('%s - averages from %i repetitions')\n" % (proj, seeds))
    str += ("    plt.margins(0.01)\n")
    str += ("    plt.ylim(0.62, 0.72)\n")
    str += ("    plt.xlabel('Number of initial base predictors')\n")
    str += ("    plt.ylabel('%s')\n" % metric)
    str += ("    plt.legend(loc=2, prop={\'size\':7})\n")
    str += ("    pdf.savefig()\n    plt.close()\n\n") #after each plot 
    return str

# # # # #

def generate_script():
    header = ("import numpy as np\n"
              "from matplotlib.backends.backend_pdf import PdfPages\n"
              "import matplotlib.pyplot as plt\n\n"
              "with PdfPages('%s.pdf') as pdf:\n\n" % (proj))
    script_name = "%s/%s/%s.py" % (project_path, directory, proj)
    with open(script_name, "w+") as plot_file:
	plot_file.write(header)
	plot_file.write(write_plot())
    plot_file.close()
    print script_name


print "\nStarting. . ."

# ensure project directory exists
project_path = abspath(argv[1])
assert exists(project_path)
directory    = 'PLOTS'
proj         = project_path.split('/')[-1]
dirnames     = sorted(filter(isdir, glob('%s/weka.classifiers.*' % project_path)))

# load and parse project properties
p            = load_properties(project_path)
fold_count   = int(p['foldCount'])
seeds        = int(p['seeds'])
metric       = p['metric']
RULE         = p['RULE']
CES_start    = '1'  #initialize ensemble with top model
max_num_clsf = len(dirnames) * seeds
sizes        = range(1,max_num_clsf+1)

#rl
strategies   = ['greedy', 'pessimistic', 'backtrack']
conv_iters   = int(p['convIters'])
age          = int(p['age'])
epsilon      = p['epsilon']
exit         = 0
algo         = 'Q'
start_state  = '0' #start randomly ('best' also an option, see rl/run.py)

#plot
x_ticks      = range(1, max_num_clsf+1, 1)
steps        = [step-1 for step in x_ticks]
color        = ['green', 'aqua', 'navy']
if not exists('%s/%s/' % (project_path, directory)):
    makedirs('%s/%s/' % (project_path, directory))

generate_script()

print "Done!\n"



