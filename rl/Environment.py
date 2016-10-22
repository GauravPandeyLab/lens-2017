#!/usr/bin/env python
"""
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see [http://www.gnu.org/licenses/].
    Inspiration: https://github.com/studywolf/blog/tree/master/RL
"""

import sys, re, operator, math, time
from numpy import random, setdiff1d
from itertools import chain, combinations
from os.path import exists
from glob import glob
from collections import OrderedDict
from pandas import DataFrame, concat, read_csv
from utilities import fmax_score, get_bps, bps2string, get_path_bag_weight

class Agent:
    def updatePosition(self, pos):  #, pos=None):
        self.pos = pos  
        self.explored = set()
        self.ens = (-1,)      

    def setWorld(self, world):
        self.world = world

    def calcState(self):
        return self.pos

    def possibleActions(self):
        return self.world.possible_actions(self.pos)

    def getEnsPerf(self):
        selected_ens = self.getEns()
        return self.world.calc_perf(selected_ens)

    # return True if successfully moved in that direction
    def goInDirection(self, dir):
        target = dir
        self.pos = dir
        return True

    def findPath(self, start_vertex, end_vertex, path=[]):
        path = path + [start_vertex]
        if start_vertex == end_vertex:
            return path
        actions = self.world.possible_actions(start_vertex)
        visited = self.getVisitedStates()
        explored = self.getExploredStates()
        valid = [element for element in actions if element in visited]
        possib = [self.ai.getQ(start_vertex, a) for a in valid]
        if len(possib) > 0:
            maxV = max(possib)
            for a in valid:
                if self.ai.getQ(start_vertex, a) == maxV:
                    extd_path = self.findPath(a, end_vertex, path)
        else:
            return path
        return extd_path
    
    def getVisitedStates(self):
        visited = []
        visited.append(self.world.start_node)
        for key, value in self.ai.q.iteritems():
            state, action = key
            visited += tuple([action])
        return set(visited)

    def getExploredStates(self):
        return self.explored

    def getPolicy(self):
        return self.findPath(self.world.start_node, self.world.exit_node, [])

    def getEns(self):
        ens = self.world.start_node
        path = self.getPolicy()
        for a in path:
            if self.world.perf[a] > self.world.perf[ens]:
                ens = a
        self.ens = ens
        return ens

    def utility(self):
        path = self.getPolicy()
        utility = []
        for i in range(0, len(path)-1):
            utility += [self.ai.getQ(path[i], path[i+1])]
        return utility  

    def getTestPerf(self, node):
        test_perf = -1
        if node == (0,): # probably RL_pessimistic was trained for too little time and all ensembles of size 2 are performing worse than the individual base predictors
            force_picks = [val for val in [tuple([y]) for y in range(1, self.world.np+1)] if val in self.getExploredStates()]
            select_from = {x:self.world.perf[x] for x in force_picks}
            index = max(select_from, key=select_from.get)[0]             
            node = tuple([index])
            test_pred = self.get_fold_probs(index, 'test', 'prediction') * self.world.bps_weight[index]
        elif (len(node) == 1):
            index = node[0]
            test_pred = self.get_fold_probs(index, 'test', 'prediction') * self.world.bps_weight[index]
        else:
            index = list(node)[0]
            test_pred = self.get_fold_probs(index, 'test', 'prediction') * self.world.bps_weight[index]  
            for index in list(node)[1:]:
                test_pred = test_pred.add(self.get_fold_probs(index, 'test', 'prediction') * self.world.bps_weight[index])
        denom       = sum([self.world.bps_weight[index] for index in list(node)])
        test_pred   = test_pred/denom
        test_labels = self.get_fold_probs(node[0], 'test', 'label') 
        test_perf   = fmax_score(test_labels, test_pred)
        return test_perf         


    def get_fold_probs(self, model_idx, set, col):
        assert set in ['valid', 'test']
        assert col in ['label', 'prediction']
        path, bag, x   = get_path_bag_weight(self.world.bps[model_idx])
        test_df     = read_csv('%s/%s-b%i-f%i-s%i.csv.gz' % (path, set, bag, self.world.fold, self.world.seed), skiprows = 1, compression = 'gzip')
        test_pred   = test_df[col]	    
        return test_pred

    def getBestBPPerf_onTest(self):
        best_bp = self.world.getBestBP()
        bp_test = self.getTestPerf(best_bp)
        return bp_test

    def getFEPerf_onTest(self):
        fe_node = tuple(list(range(1, self.world.np+1)))
        fe_test = self.getTestPerf(fe_node)
        return fe_test

class World:
    def __init__(self, np, starting_point, exit_val, input_dir, seed, fold, rule, metric):
        self.np          = np
        self.start_node  = tuple([-1])
        self.input_dir 	 = input_dir
        self.seed        = seed
        self.fold        = fold
        self.metric      = metric
        self.graph       = {}
        self.perf        = {}
        self.bps         = {}
        self.bps_weight  = {}
        self.predictors  = {}
        self.bps_weighted_pred_df = DataFrame(columns=range(0, (np+2))) #already weighted! (#cumulative)
        self.cwan        = () #cumulative weighted average numerator
        self.initialize_bps(rule)
        
        #connecting start_node with all the individual base predictors
        if starting_point == '0':
            self.start_node = tuple([0])
            self.perf[self.start_node] = float(0)
            bps = []
            for i in range(1, np+1):
                bps.append(tuple([i])) 
            self.graph[self.start_node] = bps
        else:
            self.start_node = max(self.perf, key=self.perf.get)
            self.cwan = self.start_node
            self.bps_weighted_pred_df[self.np+1] =  self.bps_weighted_pred_df[self.start_node[0]] 
            self.graph[self.start_node] = []

        #connecting the full ensemble with the exit_node
        self.exit_node = tuple(['exit'])
        self.perf[self.exit_node] = exit_val
        self.graph[tuple(list(range(1, np+1)))] = [self.exit_node]

    def calc_perf(self, node):
        if (node not in self.perf):
	    if (len(node) > len(self.cwan) and len(setdiff1d(node, self.cwan)) == 1 and node != ['exit']):
                start_time = time.time()
                bp2add = setdiff1d(node, self.cwan)
                bp = bp2add[0] 
                self.bps_weighted_pred_df[(self.np+1)] = self.bps_weighted_pred_df[(self.np+1)].add(self.bps_weighted_pred_df[bp])
                self.cwan = node
                
                denom = sum([self.bps_weight[index] for index in list(node)])
                y_score = self.bps_weighted_pred_df[(self.np+1)]/denom # new y_score variable because col_np+1 needs to be the numerator
                performance = fmax_score(self.bps_weighted_pred_df['label'], y_score)
                self.perf[node] = performance
                 
                if (len(node) == self.np):
                    self.reset_cwan_col()
                return performance
            else:  #de novo...
                start_time = time.time()
                index = list(node)[0]
                y_score = self.bps_weighted_pred_df[index]
                for index in list(node)[1:]:
                    y_score = y_score.add(self.bps_weighted_pred_df[index])
                self.bps_weighted_pred_df[(self.np+1)] = y_score
                self.cwan = node

                denom = sum([self.bps_weight[index] for index in list(node)])
                y_score = y_score/denom
                performance = fmax_score(self.bps_weighted_pred_df['label'], y_score)
                self.perf[node] = performance
                return performance
        else:
            return self.perf[node]
            

    def reset_cwan_col(self):
        if (self.start_node) == (0,):
            self.bps_weighted_pred_df[(self.np+1)] = 0
            self.cwan = ()
        else:
            self.bps_weighted_pred_df[self.np+1] =  self.bps_weighted_pred_df[self.start_node[0]]
            self.cwan = self.start_node

    def getBestBP(self):
        return (max(self.bps_weight, key=self.bps_weight.get),)

    def initialize_bps(self, rule):
        self.predictors, self.bps_weight = get_bps(self.input_dir, self.seed, self.metric, self.np)       
        for i in range(len(self.predictors)):
            index = i + 1
            self.perf[tuple([index])] = float(self.predictors[i].split(",")[1])
            self.bps[index] = self.predictors[i]
            path, bag = get_path_bag_weight(self.predictors[i])[:2]
            valid_df	= read_csv('%s/valid-b%i-f%i-s%i.csv.gz' % (path, bag, self.fold, self.seed), skiprows = 1, compression = 'gzip')
            valid_label = valid_df['label']
            valid_pred	= valid_df['prediction']
            self.bps_weighted_pred_df[index] = valid_pred * self.bps_weight[index] 
        self.bps_weighted_pred_df[0] = valid_label
        self.bps_weighted_pred_df.rename(columns={0:'label'}, inplace=True) 
        
    def possible_actions(self, pos):
        if pos == tuple([0]) or pos == tuple(list(range(1, self.np+1))):
            return self.graph[pos]
        else:
            predictors = range(1, self.np+1)
            possibilities = [(pos + (i,)) for i in predictors if i not in pos]
            pos_act = [tuple(sorted(p)) for p in possibilities]
            return pos_act
 
    def printGraph(self):
        print "\nworld.graph:"
        for key, value in sorted(self.graph.items()):
            print '\tNode %r --> %r' % (key, value)
        print "* * * * *\n"

    def printPerf(self):
        print "\nworld.perf:"
        for key, value in sorted(self.perf.items()):
            print '\t%s[%r] = %.6f' % (self.metric, key, value)
        print "* * * * *\n"

    def printClassifiers(self):
        print "\nworld.bps:"
        for key, value in sorted(self.bps.items()):
            print '\tNode %r :: %s' % (key, value)
        print "* * * * *\n"

    def getGraph(self):
        return self.graph

