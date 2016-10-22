from optparse import OptionParser
from os import getcwd, system, mkdir
from os.path import abspath, exists
from time import sleep, time

timestamp = time()
parser = OptionParser()
parser.add_option('-c', '--cores', dest = 'cores', type = 'int')
parser.add_option('-w', '--walltime', dest = 'walltime')
parser.add_option('-a', '--allocation', dest = 'allocation', default = 'TO_BE_ASSIGNED_BY_USER')
parser.add_option('-q', '--queue', dest = 'queue', default = 'expressalloc')
parser.add_option('-n', '--name', dest = 'name', default = timestamp)
(options, args) = parser.parse_args()
cores = options.cores
walltime = options.walltime
queue = options.queue
job_name = options.name
allocation = options.allocation

# make sure the logs subdirectory exists
working_dir = getcwd()
if not exists('%s/logs' % working_dir):
    mkdir('%s/logs' % working_dir)

# set the log filenames
stdout_fn = abspath('%s/logs/%s.out.txt' % (working_dir, timestamp))
stderr_fn = abspath('%s/logs/%s.err.txt' % (working_dir, timestamp))

# build the final command and run
cmd = ' '.join(args)
qsub_cmd = 'echo \"%s\" | bsub -o %s -e %s -q %s -n %i -W %s -J %s -P %s -m manda -M 8000 -R \"rusage[mem=4000]\"' % (cmd, stdout_fn, stderr_fn, queue, cores, walltime, job_name, allocation)
print qsub_cmd
system(qsub_cmd)
sleep(60)
# 10 sec of delay 
# polite delay in case called via external loop
