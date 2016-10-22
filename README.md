#lens: learning ensembles using reinforcement learning
**lens** is a customizable and extensible pipeline for learning supervised heterogeneous ensembles using techniques from the area  of reinforcement learning (RL) [Sutton and Barto, 1998]. It also evaluates other commonly used methods, including CES (Caruana's [Caruana *et al.*, 2004] Ensemble Selection algorithm). This pipeline was developed to assist research by Ana Stanescu and Gaurav Pandey (see [[Stanescu and Pandey, 2017](extra/PSB_2017_ana.pdf)]) with the support of the [Icahn Institute for Genomics and Multiscale Biology](http://icahn.mssm.edu/research/genomics) at [Mount Sinai](http://www.mountsinai.org/).

Instructions for the installation of the prerequisites and their dependencies can be followed [here](http://github.com/shwhalen/datasink). The generation of the base models requires Java, [Groovy](http://groovy.codehaus.org), and [Weka](http://www.cs.waikato.ac.nz/ml/weka/arff.html) (3.7.10). The ensemble learning scripts, including the RL module, are implemented in [Python](http://www.python.org) (2.7.6) and use several other packages, such as [pandas](http://pandas.pydata.org) (0.12) and [scikit-learn](http://scikit-learn.org) (0.12). Other versions of these libraries may also work. Similar to [datasink](http://github.com/shwhalen/datasink), the **lens** pipeline requires a training dataset, a configuration file, and a list of classification algorithms.


### Setting up the project:

    git clone https://github.com/anakstate/lens.git

We will use in our example the diabetes dataset from the [UCI repository](https://archive.ics.uci.edu/ml/datasets.html) and will download the file (in [.arff](http://www.cs.waikato.ac.nz/ml/weka/arff.html) format) into a newly created directory named /data. 

    mkdir data; cd data
    curl -O http://repository.seasr.org/Datasets/UCI/arff/diabetes.arff

The pipeline uses classification algorithms from [Weka](http://www.cs.waikato.ac.nz/ml/weka/arff.html) for base predictor generation. We use the terms *base predictor* and *model* interchangeably. The classification algorithms and their parameters are specified in the **classifiers.txt** file. The file contains all the classification algorithms used in our paper [[Stanescu and Pandey, 2017](extra/PSB_2017_ana.pdf)]. However, in this demo we will only be using five (to save time). The lines preceded by the comment marker # will be skipped and the corresponding algorithms will not be run. 

    cat > classifiers.txt << EOF
    weka.classifiers.bayes.NaïveBayes -D
    weka.classifiers.functions.Logistic -C -M 250
    #weka.classifiers.functions.MultilayerPerceptron -H o -D
    #weka.classifiers.functions.SGD -F 1
    weka.classifiers.functions.SimpleLogistic -A
    #weka.classifiers.functions.SMO -C 1.0 -M -K weka.classifiers.functions.supportVector.PolyKernel
    #weka.classifiers.functions.VotedPerceptron -I 10
    #weka.classifiers.lazy.IBk -I -K 10
    weka.classifiers.meta.AdaBoostM1
    #weka.classifiers.meta.ClassificationViaRegression
    #weka.classifiers.meta.LogitBoost -W weka.classifiers.trees.M5P
    #weka.classifiers.rules.JRip
    #weka.classifiers.rules.PART
    #weka.classifiers.trees.DecisionStump
    #weka.classifiers.trees.J48
    #weka.classifiers.trees.LMT -A -W 0.05
    weka.classifiers.trees.RandomForest -I 500 -num-slots 1
    #weka.classifiers.trees.RandomTree -N 10
    EOF

We are now creating the project directory, where all the generated subdirectories/files/results will be placed. 

    mkdir diabetes; cd diabetes

The project directory also contains a configuration file that specifies several settings for the pipeline.

    cat > config.txt << EOF
    classifiersFilename = (absolute/path/to)/data/classifiers_diabetes.txt
    inputFilename = (absolute/path/to)/data/diabetes.arff
    classifierDir = (absolute/path/to)/diabetes/
    classAttribute = class
    predictClassValue = tested_positive
    otherClassValue = tested_negative
    balanceTraining = true
    balanceTest = false
    balanceValidation = false
    foldCount = 5
    writeModel = true
    seeds = 2
    bags = 2
    metric = fmax 
    RULE = WA
    useCluster = n
    convIters = 0
    age = 100
    epsilon = 0.01 
    EOF


### Demo: Running the pipeline

The **lens** pipeline is designed for multicore and distributed environments, such that many processes can be run simultaneously, taking advantage of all the machine's available cores. The cross-validation technique can be straightforwardly parallelized as the tasks are independent of each other for each fold, therefore the processes are spawned and each writes its output to a unique file. We start by training the classification algorithms and then using the produced models to classify the validation and test sets. The models produced by the algorithms can be stored (writeModel = true), with the caveat that disk space usage will substantially increase. Let's start:

	cd lens
	python step1_generate.py absolute/(or/relative/)path/to/diabetes

As specified in the **config.txt file**, the data is first divided into five folds (foldCount = 5) of independent training (60% of the entire dataset), validation (20%), and test (20%) splits for cross validation. Each training split is resampled with replacement two times (bags = 2) via a process called bagging ([Breiman1996]), resulting in ten (#classification algorithms * bags) base predictors. The validation set helps construct the CES/RL-based supervised ensembles. The test split will be used to assess the actual performance of the ensemble selection technique. All experiments will be repeated two times (seeds = 2), using different seeds for shuffling the original data (diabetes.arff). A visual description of the workflow can be seen [here](figures/Flowchart.png). For the **classifiers.txt** file in the example, the following folders will be created inside the project directory (*i.e.*, diabetes/): 

	weka.classifiers.bayes.NaïveBayes
	weka.classifiers.functions.Logistic
	weka.classifiers.functions.SimpleLogistic
	weka.classifiers.meta.AdaBoostM1
	weka.classifiers.trees.RandomForest

	
Each directory (named by a classification algorithm) contains .gzip files of each generated model, the predicted probabilities of the model on the validation set, and the predicted probabilities of the model on the test set (*e.g.*, NaïveBayes-b\<i\>-f\<i\>-s\<i\>.model.gz, valid-b\<i\>-f\<i\>-s\<i\>.csv.gz, valid-b\<i\>-f\<i\>-s\<i\>.csv.gz, *etc.*, where b = bagged version, f = fold number, s = seed). Next, let's find out how the models perform (in terms of the performance metric from the **config.txt** file, metric = fmax):

	python step2_order.py absolute/(or/relative/)path/to/diabetes
	
Step2 will order all the base predictors generated in step1 based on their performance. The order files will be stored in a newly created directory named ENSEMBLES, inside the project directory. For each repetition (*i.e.*, seed) there will be an independent file containing the corresponding ordered list. As an example of what to expect, the base predictors and their performance for seed 1 are shown in detail below (diabetes/ENSEMBLES/order_of_seed1_fmax.txt): 
 
	weka.classifiers.functions.Logistic_bag0, 0.769230769231 
	weka.classifiers.functions.SimpleLogistic_bag0, 0.752136752137
	weka.classifiers.functions.SimpleLogistic_bag1, 0.745762711864
	weka.classifiers.meta.AdaBoostM1_bag0, 0.738738738739 
	weka.classifiers.bayes.NaiveBayes_bag0, 0.723076923077 
	weka.classifiers.functions.Logistic_bag1, 0.719298245614 
	weka.classifiers.trees.RandomForest_bag0, 0.717948717949 
	weka.classifiers.bayes.NaiveBayes_bag1, 0.70796460177 
	weka.classifiers.trees.RandomForest_bag1, 0.697247706422 
	weka.classifiers.meta.AdaBoostM1_bag1, 0.649350649351 


In order to study the variation of the ensemble selection algorithms with increasingly larger numbers of base predictors, we start with a set consisting of only one base predictor and increase this set gradually, with randomly selected base predictors, until we reach the entire set of ten base predictors (or models). These models are chosen randomly, without replacement, from the order file, based on their performance. We can consider three categories of models, *good*, *medium*, and *weak*, by dividing the ordered models equally between these bins. More specifically, for each size, an equal number of models from each category are selected as the initial pool of available base predictors. 

	python step3_best.py absolute/(or/relative/)path/to/diabetes
	
This step will generate the baselines of the experiments and place them in the BASELINE directory, under their corresponding seed.  First, we consider the best base predictor (BP) from each initial pool of base classifiers, which is the classifier with the highest classification prediction performance on the validation set. At the opposite end of the spectrum, the full ensemble (FE), consisting of all initial base predictors, will be the largest ensemble, and what we consider our second baseline. The baseline results of our demo for both BP and FE are grouped per seed, as shown below: 

	/diabetes/BASELINE/ORDER[0-1]/BP_bp<size>_seed[0-1].fmax (size = 1 .. #base predictors)
	/diabetes/BASELINE/ORDER[0-1]/FE_bp<size>_seed[0-1]_WA.fmax

** Note**: The ensembles selected by the various approaches we tested (FE, RL, CES) are created by combining the probabilities produced by the constituent base predictors using a weighted average (RULE = WA). Here, the importance (weight) of each base predictor is proportional to its predictive performance (fmax score from the order file) on the validation set for all the approaches. The next step runs the CES algorithm, for all pool sizes:

	python step4_CES_run.py absolute/(or/relative/)path/to/diabetes
	
The resulting ensembles are stored in the CES_OUTPUT directory, one file per fold. After the CES ensembles are determined for each of the folds, it's time to evaluate them on the corresponding test split, and then merge all the predicted probabilities in order to assess the ensemble selection algorithm's performance on the complete test set: 

	python step4_CES_ens.py absolute/(or/relative/)path/to/diabetes
	
Next, we run the RL-based approaches. The RL module is a separate directory within the code, and it is written in an object oriented fashion. Inspiration for the RL module was drawn from Travis DeWolf's [blog](https://studywolf.wordpress.com) on [RL](https://github.com/studywolf/blog/tree/master/RL). A general UML diagram of the RL module is depicted below:

<p align="center">
  <img src="figures/rl_UML.png" width="70%"><br>
  <em> UML diagram of the RL module. </em>
</p>

We define the environment as a deterministic one, in the sense that taking the same action in the same state on two different occasions cannot result in different next states and/or different reinforcement values. More specifically, the world in which our agent operates consists of all possible subsets of the n base predictors, each serving as a possible ensemble, thus consisting of 2^n states. The environment includes the empty set, which is considered the start state in our implementation. It can be viewed as a lattice, and the arrows represent the actions the agent is allowed to take at each state. Such a world is captured in the Environment.World class and an agent that can operate in this world is implemented in the Environment.Agent class.

In addition to the attributes and methods inherited from the more general class Environment.Agent, our more specialized agent (*i.e.*, our proposed ensemble selection algorithm) must also keep track of other variables, such as its previous state, its last action, what is its age at a particular time, the exploration/exploitation probability (epsilon) which controls how much exploration the agent is allowed to perform, *etc*. The agent is using the proposed search strategies (RL_greedy, RL_pessimistic, RL_backtrack) (see Section 3 of [[Stanescu and Pandey, 2017](extra/PSB_2017_ana.pdf)]) to learn how to behave in the world (*i.e.*, how to traverse the lattice and how to use rewards) in order to maximize its long-term reward. The agent cannot operate without its AI module, supported by the Q-Learning algorithm (QL). The QL algorithm requires parameters such as *alpha* (learning rate) and *gamma* (discount factor). Other algorithms (*e.g.*, SARSA) can be easily plugged-in given the architecture above. 

The goal of RL is to repeat the action-observation process (for a fixed number of steps, *i.e.*, the *age* field, or until a certain number of consecutive episodes yield the same ensemble, *i.e.*, the *conv_iters* field from the **config.txt** file), that results in the agent learning a good/optimal strategy, called 'policy', for collecting rewards, and completing the task(s) at hand. A policy in our case is a traversal path inside the lattice, from start to finish. 

Let's now run the RL-based ensemble selection approaches on each fold, just like we did for CES. The **rl** module is accessed via the driver class *rl/run.py*, called from the following script:  

	python step5_RL_run.py absolute/(or/relative/)path/to/diabetes
	
Each fold will yield its own ensemble and the predictions from all the folds will be used to obtain one performance metric that will reflect the ensemble selection algorithms' performance:

	python step5_RL_ens.py absolute/(or/relative/)path/to/diabetes

Next, step6 parses the results of all the experiments (BP, FE, CES, and the RL-based approaches) and stores them in the RESULTS directory:

	python step6_results.py absolute/(or/relative/)path/to/diabetes
	
These are the .csv files with all the results (max and ensemble dimension) obtained on the test sets, for the experiments run in the demo: 

	BEST PREDICTOR VALUES	:: /diabetes/RESULTS/BP/RESULTS_BP_fmax.csv
	FE VALUES		 		:: /diabetes/RESULTS/FE/RESULTS_FE_WA_fmax.csv
	CES VALUES		 		:: /diabetes/RESULTS/CES/RESULTS_CES_WA_start-1_fmax.csv
	CES DIMENSIONS		 	:: /diabetes/RESULTS/CES/RESULTS_CES_WA_start-1_fmax_dim.csv
	RL VALUES		 		:: /diabetes/RESULTS/RL/RESULTS_RL_epsilon0.01_pre100_conv0_exit0_greedy_WA_Q_start-0_fmax.csv
	CES DIMENSIONS		 	:: /diabetes/RESULTS/RL/RESULTS_RL_epsilon0.01_pre100_conv0_exit0_greedy_WA_Q_start-0_fmax_dim.csv
	RL VALUES		 		:: /diabetes/RESULTS/RL/RESULTS_RL_epsilon0.01_pre100_conv0_exit0_pessimistic_WA_Q_start-0_fmax.csv
	CES DIMENSIONS		 	:: /diabetes/RESULTS/RL/RESULTS_RL_epsilon0.01_pre100_conv0_exit0_pessimistic_WA_Q_start-0_fmax_dim.csv
	RL VALUES		 		:: /diabetes/RESULTS/RL/RESULTS_RL_epsilon0.01_pre100_conv0_exit0_backtrack_WA_Q_start-0_fmax.csv
	CES DIMENSIONS		 	:: /diabetes/RESULTS/RL/RESULTS_RL_epsilon0.01_pre100_conv0_exit0_backtrack_WA_Q_start-0_fmax_dim.csv

For example, the CES results files show the performance values (in terms of fmax) and the sizes (averaged over the folds) of all ensembles, for each repetition/seed:

	>  diabetes/RESULTS/CES/RESULTS_CES_WA_start-1_fmax.csv
	Seed/Order,ens1, ens2, ens3, ens4, ens5, ens6, ens7, ens8, ens9, ens10
	SEED_0,0.660714,0.674923,0.675926,0.667582,0.675676,0.694943,0.692557,0.692683,0.692683,0.687296
	SEED_1,0.655172,0.677326,0.663333,0.676923,0.683969,0.688207,0.691318,0.693291,0.69102,0.688331

	>  diabetes/RESULTS/CES/RESULTS_CES_WA_start-1_fmax_dim.csv
	Seed/Order,ens1, ens2, ens3, ens4, ens5, ens6, ens7, ens8, ens9, ens10
	SEED_0,1.0,2.0,2.2,2.6,2.8,2.8,3.2,3.4,3.4,3.4
	SEED_1,1.0,2.0,2.4,2.4,2.6,2.8,3.2,4.0,3.8,4.2

Since it is rather difficult to assess all the results from the .csv files for all experiments, step7 will create a python script (**[diabetes.py](extra/diabetes.py)**) that generates a line chart: 

	python step7_plot.py absolute/(or/relative/)path/to/diabetes

To obtain the plot, simply run the script created at step7 to generate a pdf file (**[diabetes.pdf](extra/diabetes.pdf)**):

	 python diabetes.py
	 
The performances (on the y-axis) represent averages over the repetitions performed (seeds), and the lines are annotated with the average sizes (first over the folds and secondly over the repetitions) of the ensembles learnt by the various ensemble selection algorithms at the points shown on the x-axis. We used the area under these curves, denoted auESC (area under Ensemble Selection Curve), as a global evaluation measure for the various algorithms in our study, as it provides a global assessment of ensemble performance across a variety of base predictors. The area is normalized by its maximum possible value, which is the total number of base predictors (the maximum possible value of fmax is one); thus, the maximum value of auESC is one. However, this metric does not follow the same characteristics as auROC, such as random predictors/ensembles producing a score of 0.5. Therefore, auESC is mostly intended for comparative analyses between algorithms running on the same datasets (as done in our experiments), rather than for assessing the absolute performance of these algorithms.


===============
## Bibliography:

- [Stanescu, A., and Pandey, G. "Learning parsimonious ensembles for unbalanced computational genomics problems." Pacific Symposium on Biocomputing (2017)](extra/PSB_2017_ana.pdf)
- Sutton, R. S., and A. G. Barto. "Introduction to Reinforcement Learning." 1st. ed. (1998)
- Radivojac, P., Clark, W.T., Oron, T.R., Schnoes, A.M., Wittkop, T., Sokolov, A., Graim, K., Funk, C., Verspoor, K., Ben-Hur, A. and Pandey, G. "A large-scale evaluation of computational protein function prediction." Nature methods (2013)
- Hall, M., Frank, E., Holmes, G., Pfahringer, B., Reutemann, P. and Witten, I.H. "The WEKA data mining software: an update." ACM SIGKDD explorations newsletter (2009)
- Gansner, E. R., and North, S. C. "An open graph visualization system and its applications to software engineering." Software Practice and Experience (2000)
- Caruana, R., Munson, A. and Niculescu-Mizil, A. "Getting the most out of ensemble selection." Proceedings of the Sixth International Conference on Data Mining (2006) 
- Caruana, R., Niculescu-Mizil, A., Crew, G., and Ksikes, A. "Ensemble Selection from Libraries of Models." Proceedings of the 21st International Conference on Machine Learning (2014)
- Breiman, L. "Bagging predictors." Machine learning (1996)

