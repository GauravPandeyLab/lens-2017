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
import Environment, QL
import collections, numpy, sys, time
from collections import OrderedDict
from pandas import DataFrame, concat, read_csv



class Agent(Environment.Agent):

    def __init__(self, epsilon, seed, strategy, age, conv):
        self.explored = set([])
        self.ai = QL.QL(epsilon, seed, alpha=0.1, gamma=0.9)
        self.last_action = None
        self.max = 0
        self.previous_state = None
        self.strategy = strategy
        self.age = age
        self.conv = conv
        self.predictions_df = []
        self.num_episodes = -1

    def train(self): # for a fixed number of steps, not episodes!
        self.pos = self.world.start_node
        method = 'RL_' + self.strategy + "()"
        for i in range(int(self.age)+1):
            exec('self.' + method)

    def converge(self, iterations): # converge after a certain number of consecutive episodes yield the same result
        self.pos = self.world.start_node
        consecutive_count = 0
        episodes = 0
        np = self.world.np
        method = 'RL_' + self.strategy + "()"
        self.train() #pre-training
        self.current = self.getEns()
        while (consecutive_count < iterations):
            exec('self.' + method)
            self.age += 1
            if (self.pos == self.world.exit_node): 
                episodes += 1
                next = self.getEns()
                if (self.current == next): 
                    consecutive_count += 1
                else: 
                    consecutive_count = 0 
                    self.current = next
        self.num_episodes = episodes
              
  
    def RL_greedy(self):
        state = self.calcState()
        if (state == self.world.exit_node):
            self.pos = self.world.start_node
            self.last_action = None
            self.max = 0
        else:
            actions = self.possibleActions()
            action  = self.ai.chooseAction(state, actions)
            self.explored.add(action)
            if state not in self.world.graph:
                self.world.graph[state] = []
            if action not in self.world.graph[state]:
                self.world.graph[state].append(action)
            # we want to reward the agent when it is leaving one of the classifiers, not when leaving the start_state (when the start_state = 0)
            reward = (0 if (len(action) == 1 and self.world.start_node == (0,)) else self.calc_reward_performance(action))
            if reward > self.max:
                self.max = reward
            else:
               reward = self.max    
            if self.last_action is not (None): 
                self.ai.learnQL(self.lastState, self.last_action, reward, state, actions)
            self.lastState = state
            self.last_action = action
            self.goInDirection(action)

    def RL_pessimistic(self):
        state = self.calcState()
        if (state == self.world.exit_node):
            self.pos = self.world.start_node
            self.last_action = None
        else:
            actions = self.possibleActions()
            action = self.ai.chooseAction(state, actions)
            self.explored.add(action)
            reward = (0 if (len(action) == 1 and self.world.start_node == (0,)) else self.calc_reward_diff(action))
            if reward < 0: # if performance gets decreased by the action
                self.pos = self.world.start_node # go to start
                self.last_action = None
            else:
                if self.last_action is not (None): 
                    self.ai.learnQL(self.lastState, self.last_action, reward, state, actions)
                self.lastState = state
                self.last_action = action
                self.goInDirection(action)  

    def RL_backtrack(self):
        state = self.calcState()
        if (state == self.world.exit_node):
            self.pos = self.world.start_node
            self.previous_state = None
            self.last_action = None
        else:
            actions = self.possibleActions()
            action = self.ai.chooseAction(state, actions)
            self.explored.add(action)
            reward = (0 if (len(action) == 1 and self.world.start_node == (0,)) else self.calc_reward_diff(action))
            if reward < 0: # if performance gets decreased by the action
                if self.last_action is not (None): # go to previous state
                    self.pos = self.previous_state
            else:
                if self.last_action is not (None):
                    self.ai.learnQL(self.lastState, self.last_action, reward, state, actions)
                self.lastState = state
                self.last_action = action
                self.previous_state = state
                self.goInDirection(action)

        
    def calc_reward_diff(self, action): 
        pos0 = (0 if self.pos == (0,) else self.world.calc_perf(self.pos))
        pos1 = self.world.calc_perf(action)
        return (pos1-pos0)

    def calc_reward_performance(self, action):
        return self.world.calc_perf(action)



