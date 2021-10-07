#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 15:11:08 2021

@author: zen
"""
import learner as ln
import numpy as np

class SPData(ln.Data):
    '''Data for solving the shortest path with obstacles
    '''
    def __init__(self, train_num, test_num, train_traj, test_traj, train_noise, test_noise, q_terminal, t_terminal, q_initial):
        super(SPData, self).__init__()
        self.train_num = train_num
        self.test_num = test_num
        self.train_traj = train_traj
        self.test_traj = test_traj
        self.train_noise = train_noise
        self.test_noise = test_noise
        self.q_terminal = q_terminal
        self.t_terminal = t_terminal
        self.q_initial = q_initial
        self.__init_data()
    
    '''Here X, y represent the input and output of a neural network, for coordinates
    we use p,q,P,Q instead
    '''
    def generate(self, num, traj_num, noise = 0):
        X, y = {}, {}
        X['interval'] = np.linspace(0, self.t_terminal, num)[:,None]
        # positions
        X['bd'] = np.array([0, self.t_terminal])[:,None]
        # values (could be changed if needed)
        q_initial = np.expand_dims(self.q_initial, 0).repeat(traj_num, axis = 0)
        q_initial = q_initial + (2 * np.random.rand(*q_initial.shape) - 1) * noise
        q_terminal = np.expand_dims(self.q_terminal, 0).repeat(traj_num, axis = 0)
        y['bd'] = np.transpose(np.array([q_initial, q_terminal]), (1,0,2))
        return X, y
    
    def __init_data(self):
        self.X_train, self.y_train = self.generate(self.train_num, self.train_traj, self.train_noise)
        self.X_test, self.y_test = self.generate(self.test_num, self.test_traj, self.test_noise)