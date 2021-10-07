#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 11:06:05 2021

@author: zen
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import learner as ln
from data import SPData
from nn_test12 import SPNN
from PS_method import PSmethod
import os
from time import perf_counter
        
class MZNN(SPNN):
    '''NN for solving the optimal control of shortest path in mazes
    '''
    def __init__(self, dim, layers, width, activation, ntype, dr = 0.2, ws = 0, angles = 0.5,
                 qr = 0.1, ql = 1, l = 0.001, eps = 0.1, lam = 1, C = 25, add_dim = 0,
                 ifpenalty = True, rho = 1.0, add_loss = 0, update_lagmul_freq = 10, trajs = 1, 
                 dtype = None, device = None):
        super(MZNN, self).__init__(dim, layers, width, activation, ntype, l, eps, lam, C, add_dim,
        ifpenalty, rho, add_loss, update_lagmul_freq, trajs, dtype, device)
        self.dr = dr                # radius of the drone
        self.ws = ws                # starting points of the obstacles
        self.angles = angles        # angles of the obstacles
        self.num_obs = len(ws)      
        self.qr = qr                # width of the obstacle
        self.ql = ql                # length of the obstacle
    
    # square of distance between q and the line segment connected by w, v, then minus a constant
    def dist_sq(self, w, v, q, ql):
        l2 = ql[None,:,None] ** 2
        q = q.reshape([-1, 1, 2])
        d = torch.sum((q - v[None,...]) * (w - v), dim = -1, keepdims = True) / l2
        # t is the truncation of d into [0,1]
        t = 1.0 - torch.relu(1.0- torch.relu(d))
        projection = v[:,None,:] + torch.transpose(t, 0, 1) @ (w - v)[:,None,:]
        d = torch.sum((q.squeeze() - projection) ** 2, dim = -1, keepdims = True)
        d = d.reshape([self.num_obs, -1, self.dim // 2, 1])
        d = torch.transpose(d, 1, 2).reshape([self.num_obs * self.dim // 2, -1, 1]) - (self.qr + self.dr)**2
        return d
    
    # try to avoid for loop in this function
    # return dim: #constraints * (num pts in all trajs) * 1
    def h(self, q):
        ws = torch.tensor(self.ws, device = self.device, dtype = self.dtype)
        ql = torch.tensor(self.ql, device = self.device, dtype = self.dtype)
        angles = torch.tensor(self.angles, device = self.device, dtype = self.dtype)
        vs = ws + torch.stack([torch.cos(angles), torch.sin(angles)], dim = -1) * ql[:,None]
        h = self.dist_sq(ws, vs, q, ql)
        x = q.reshape([-1, self.dim // 2, 2, 1])
        x = torch.transpose(x, 1, 2)
        y = torch.transpose(x, 2, 3)
        z = torch.sum((x - y)**2, dim = 1)
        mask = ~torch.eye(z.shape[1],z.shape[2], dtype=bool)
        min_value = z[:, mask].t()[...,None] - (self.dr * 2) **2
        h = torch.cat([min_value, h], dim = 0)
        return h
    
    def dist_sq_np(self, w, v, q, ql):
        l2 = ql[None,:,None] ** 2
        q = q.reshape([-1, 1, 2])
        d = np.sum((q - v[None,...]) * (w - v), axis = -1, keepdims = True) / l2
        t = np.maximum(np.zeros_like(d), np.minimum(np.ones_like(d), d))
        projection = v[:,None,:] + np.transpose(t, (1, 0, 2)) @ (w - v)[:,None,:]
        d = np.sum((q.squeeze() - projection) ** 2, axis = -1, keepdims = True)
        d = d.reshape([self.num_obs, -1, self.dim // 2, 1])
        d = np.transpose(d, (0,2,1,3)).reshape([self.num_obs * self.dim // 2, -1, 1]) - (self.qr + self.dr)**2
        return d
    
    def h_np(self, q):
        ws = np.array(self.ws)
        ql = np.array(self.ql)
        angles = np.array(self.angles)
        vs = ws + np.stack([np.cos(angles), np.sin(angles)], axis = -1) * ql[:,None]
        h = self.dist_sq_np(ws, vs, q, ql)
        x = q.reshape([-1, self.dim // 2, 2, 1])
        x = np.transpose(x, (0,2,1,3))
        y = np.transpose(x, (0,1,3,2))
        z = np.sum((x - y)**2, axis = 1)
        mask = ~np.eye(z.shape[1],z.shape[2], dtype=bool)
        min_value = (np.transpose(z[:, mask])[...,None]) - (self.dr * 2)**2
        h = np.concatenate([min_value, h], axis = 0)
        return h
    
def update_lag_mul(data, net):
    net.update_lag_mul(data.X_train['interval'], data.X_train['bd'], data.y_train['bd'])
    # output mean cost over several trajectories
    cost, hmin = compute_cost_hmin(data.X_train['interval'], net, net.trajs)
    return cost, hmin
           
def compute_cost_hmin(t, net, traj_count):
    cost = torch.mean(net.value_function(t)[:traj_count]).item()
    hmin = net.hmin_function(t, traj_count).item()
    print('cost value: {}\t'.format(cost))
    print('constraint value: {}\n'.format(hmin), flush=True)
    return cost, hmin

def dist(w, v, p, qr):
    l2 = np.sum((w - v) ** 2)
    d = np.sum((p - v) * (w - v), axis = -1, keepdims = True) / l2
    t = np.maximum(np.zeros_like(d), np.minimum(np.ones_like(d), d))
    projection = v + np.matmul(t, (w - v))
    return np.sqrt(np.sum((p - projection) ** 2, axis = -1)) - qr

def plot_heat(q_pred, net, filename, num_interpolate, num_total, y_train, y_test):
    num_drone = net.dim // 2
    x = np.linspace(-5,5,100)
    y = np.linspace(-5,5,100)
    xx, yy = np.meshgrid(x, y)
    zz = []
    for i, w in enumerate(net.ws):
        w = np.array([w])
        v = w + np.array([[np.cos(net.angles[i]), np.sin(net.angles[i])]]) * net.ql[i]
        h = lambda p: dist(w, v, p, net.qr)
        p = np.concatenate([xx.reshape([-1,1]), yy.reshape([-1,1])], axis = -1)
        zz.append(h(p).reshape([100,100]))
    zz = np.min(np.array(zz), axis = 0)
    
    plt.figure(figsize = [4*num_interpolate,4*2])
    for index in range(num_total):
        ax = plt.subplot(2,num_interpolate,index+1)
        ax.set_xlim((-5, 5))
        ax.set_ylim((-5, 5))
        plt.contour(xx,yy,zz,[0])
        for i in range(y_train['bd'].size()[0]):
            for j in range(num_drone):
                ax.add_patch(plt.Circle((y_train['bd'][i, 0, 2*j], y_train['bd'][i, 0, 2*j+1]), net.dr, fill = True, color = 'yellow', linestyle = '--'))
        for i in range(num_drone):
            ax.add_patch(plt.Circle((y_test['bd'][index,0,2*i], y_test['bd'][index,0,2*i+1]), net.dr, fill = False, color = 'black', linestyle = '--'))
            ax.add_patch(plt.Circle((y_test['bd'][index,1,2*i], y_test['bd'][index,1,2*i+1]), net.dr, fill = False, color = 'red', linestyle = '-'))
            plt.plot(q_pred[index,:,2*i], q_pred[index,:,2*i+1])
    plt.savefig('figs/'+filename+'.pdf')
    

def plot_anime(q_pred, net, filename):
    from matplotlib import animation
    from random import randint
    num_drone = net.dim // 2
    x = np.linspace(-5,5,100)
    y = np.linspace(-5,5,100)
    xx, yy = np.meshgrid(x, y)
    zz = []
    for i, w in enumerate(net.ws):
        w = np.array([w])
        v = w + np.array([[np.cos(net.angles[i]), np.sin(net.angles[i])]]) * net.ql[i]
        h = lambda p: dist(w, v, p, net.qr)
        p = np.concatenate([xx.reshape([-1,1]), yy.reshape([-1,1])], axis = -1)
        zz.append(h(p).reshape([100,100]))
    zz = np.min(np.array(zz), axis = 0)
    fig = plt.figure(figsize = [4,4])
    ax = plt.axes(xlim=(-5, 5), ylim=(-5,5))
    plt.contour(xx,yy,zz,[0])
    colors = []
    for i in range(num_drone):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
    drones = []
    for i in range(num_drone):
        drones.append(plt.Circle((q_pred[0, 2*i], q_pred[0, 2*i+1]), net.dr, fill = False, color = colors[i]))
    def init():
        for i in range(num_drone):
            ax.add_patch(drones[i])
        return drones 
    def animate(i):
        for j in range(num_drone):
            drones[j].center = (q_pred[i,2*j], q_pred[i,2*j+1])
        return drones
    frames = q_pred.shape[0]
    delay_time = 20000 // q_pred.shape[0]
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=frames, interval=delay_time, blit=True, repeat = False)
    anim.save('figs/'+filename+'.gif', writer='imagemagick', fps=30)
    plt.show()
    plt.close()
    
def plot_cost_constraint(data, net, loss, filename, print_every):
    cost = loss[:,-2]
    hmin = loss[:,-1]
    iterations = np.arange(0, print_every * len(cost), print_every)
    plt.figure()
    plt.plot(iterations, cost, label = 'cost', c = 'b')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figs/'+filename+'/cost.pdf')
    plt.show()
    plt.close()
    plt.figure()
    plt.plot(iterations, hmin, label = 'constraint', c = 'r')
    plt.plot(iterations, 0*hmin, '.', markersize = 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig('figs/'+filename+'/constraint.pdf')
    plt.close()

# q_pred is of size num_times * dim
def plot_fig(q_pred, net, filename, drone_time_index, color_list, y_train):
    num_drone = net.dim // 2
    x = np.linspace(-5,5,100)
    y = np.linspace(-5,5,100)
    xx, yy = np.meshgrid(x, y)
    zz = []
    for i, w in enumerate(net.ws):
        w = np.array([w])
        v = w + np.array([[np.cos(net.angles[i]), np.sin(net.angles[i])]]) * net.ql[i]
        h = lambda p: dist(w, v, p, net.qr)
        p = np.concatenate([xx.reshape([-1,1]), yy.reshape([-1,1])], axis = -1)
        zz.append(h(p).reshape([100,100]))
    zz = np.min(np.array(zz), axis = 0)

    fig = plt.figure(figsize = [4,4])
    ax = plt.axes(xlim=(-5, 5), ylim=(-5,5))    
    plt.contour(xx,yy,zz,[0])
    # plot drones as circles at the position where time equals drone_time_index
    for i in range(num_drone):
        # plot training data
        for j in range(y_train['bd'].shape[0]):
            ax.add_patch(plt.Circle((y_train['bd'][j, 0, 2*i], y_train['bd'][j, 0, 2*i+1]), 0.05, fill = True, color = 'yellow'))
        ax.add_patch(plt.Circle((q_pred[drone_time_index, 2*i], q_pred[drone_time_index, 2*i+1]), net.dr, fill = False, color = color_list[i])) #, linestyle = '--'))
        plt.plot(q_pred[:drone_time_index+1,2*i], q_pred[:drone_time_index+1,2*i+1], color = color_list[i])
    plt.savefig(filename+'.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Hyperparameters to be tuned')
    parser.add_argument('--l', type = float, default = 0.004, help = 'l in the soft penalty')
    parser.add_argument('--eps', type = float, default = 0.0004, help = 'epsilon in the soft penalty')
    parser.add_argument('--lam', type = float, default = 600.0, help = 'weight of boundary loss')
    parser.add_argument('--ntype', type = str, default = 'G', help = 'type of NN')
    parser.add_argument('--layers', type = int, default = 6, help = 'layers of NN')
    parser.add_argument('--width', type = int, default = 60, help = 'width of NN')
    parser.add_argument('--act', type = str, default = 'relu', help = 'activation function')
    parser.add_argument('--lagmulfreq', type = int, default = 1, help = 'the frequency of updating lagmul')
    parser.add_argument('--rho', type = float, default = 1.0, help = 'parameter for aug Lag')
    parser.add_argument('--iters', type = int, default = 100000, help = 'number of iterations')
    parser.add_argument('--lbfgsiters', type = int, default = 1000, help = 'number of lbfgs iterations for testcase2')
    # modify the following flags to test
    parser.add_argument('--testcase', type = int, default = 1, help = '1 for comparison, 2 for mul traj, 3 for maze')
    parser.add_argument('--penalty', action="store_true", help = 'whether use penalty function')
    parser.add_argument('--addloss', type = int, default = 0, help = '0 for no added loss, 1 for log penalty or aug Lag, 2 for quadratic penalty')
    parser.add_argument('--adddim', type = int, default = 0, help = 'added dimension')
    parser.add_argument('--gpu', action="store_true", help = 'whether to use gpu')
    parser.add_argument('--seed', type = int, default = 1, help = 'random seed')
    # use this flag if loading instead of training
    parser.add_argument('--loadno', type = int, default = 100000, help = 'the no. for the model to load')
    parser.add_argument('--plotps', action="store_true", help = 'True if plot results from ps, False if from nn')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.gpu:
        device = 'gpu' # 'cpu' or 'gpu'
    else:
        device = 'cpu' # 'cpu' or 'gpu'

    # data
    t_terminal = 1.0 
    train_num = 200
    test_num = 200
    train_traj = 100
    test_traj = 100
    train_noise = 1.0
    test_noise = 0.0
    problem_name = 'door2'
    num_interpolate = 3
    traj_count = num_interpolate * 2
   
    qr = 0.5
    dr = 0.5
    ql = [1.1, 1.1]
    ws = [[-2.5, 0], [1.4, 0]]
    angles = [0, 0]
    q_initial = [-2, -2, 2, -2, 2, 2, -2, 2]
    q_terminal = [2, 2, -2, 2, -2, -2, 2, -2]
    
    dim = len(q_initial)
    add_dim = args.adddim
    rho = args.rho
    l = args.l
    eps = args.eps
    lam = args.lam
    ntype = args.ntype
    layers = args.layers
    width = args.width
    activation = args.act
    
    distsq = True

    foldername = 'testcase_{}_distsquare_'.format(args.testcase)
    foldername += 'eps_{}_l_{}_'.format(eps, l)
    if args.penalty:
        foldername += 'penalty_'
    else:
        foldername += 'auglag_'
    foldername += 'lbfgsiters_{}'.format(args.lbfgsiters)
    foldername += 'addloss_{}'.format(args.addloss)
    foldername += '_adddim_{}_'.format(add_dim)
    foldername += '_seed_{}'.format(args.seed)
    print(foldername, flush = True)
    figname = foldername

    output_foldername = 'saved_figures/test2/'
    if not os.path.exists(output_foldername): os.mkdir(output_foldername)

    data = SPData(train_num, test_num, train_traj, test_traj, train_noise, test_noise, q_terminal, t_terminal, q_initial)
    data.device = 'gpu'
    data.dtype = 'float'

    # load nn
    net = torch.load('model/'+foldername+'/model{}.pkl'.format(args.loadno), map_location=torch.device('cuda'))


    # new test data and train LBFGS
    y_train, X_test, y_test = data.y_train, data.X_test, data.y_test
    y_test['bd'] = y_test['bd'] + torch.tensor(np.concatenate([(2 * np.random.rand(data.train_traj,1,net.dim) - 1), np.zeros((data.train_traj,1,net.dim))], axis = 1), device = net.device, dtype = net.dtype)
    y_test['bd'][num_interpolate,0] = torch.tensor([-2, -4, 2, -4, 2,4, -2,4], device = net.device, dtype = net.dtype)
    y_test['bd'][num_interpolate+1,0] = torch.tensor([-4, -4, 0, -4, 0,4, -4,4], device = net.device, dtype = net.dtype)
    y_test['bd'][num_interpolate+2,0] = torch.tensor([0, -4, 0, -2, 0,2, 0,4], device = net.device, dtype = net.dtype)
    y_test['bd'] = y_test['bd'].float()
    # LBFGS training
    net.LBFGS_training(X_test, y_test, True, args.lbfgsiters)
    
    if args.plotps:
        num_times = 20
        num_nodes = 3
        PSiters = 10000
        time_endpts = np.linspace(0.0, t_terminal, num_times)
        q_pred = np.zeros((traj_count, (num_times-1) * num_nodes, dim))
        # plot initial x
        for i in range(traj_count):
            # set initialization for PS method
            x_init_ps = y_test['bd'][i,0].detach().cpu().numpy()
            x_term_ps = y_test['bd'][i,1].detach().cpu().numpy()
            PSsolver = PSmethod(time_endpts, num_nodes, net.dim, net, x_init_ps, x_term_ps, i)
            PSsolver.solve(PSiters)
            q_pred[i] = PSsolver.get_x()
    else:
        q_pred = net.predict_q(X_test['interval'], True)


    if args.plotps:
        lossname = 'ps'
    else:
        lossname = 'nn'


    num_time_slot = 4
    total_time_ind = q_pred.shape[1]
    for j in range(traj_count):
        for i in range(num_time_slot): 
            drone_time_index = (total_time_ind -1) * i // (num_time_slot-1)
            plot_fig(q_pred[j], net, output_foldername + lossname + '_traj_{}_timeindex_{}'.format(j,i), drone_time_index, color_list=['red', 'black', 'blue', 'green'], y_train = y_train)



if __name__ == '__main__':
    main()
