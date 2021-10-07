#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 20:12:56 2021

@author: zzhang99
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import learner as ln
from data import SPData
from nn import SPNN
from PS_method import PSmethod
import os
from time import perf_counter
from learner.utils import pathpatch_2d_to_3d, pathpatch_translate
        
class MZNN(SPNN):
    '''NN for solving the optimal control of shortest path in mazes
    '''
    def __init__(self, dim, layers, width, activation, ntype, barrier, dr = 0.2, l = 0.001, eps = 0.1, 
                 phy_dim = 2, lam = 1, C = 25, add_dim = 0, ifpenalty = True, rho = 1.0, add_loss = 0, 
                 update_lagmul_freq = 10, trajs = 1, dtype = None, device = None):
        super(MZNN, self).__init__(dim, phy_dim, layers, width, activation, ntype, l, eps, lam, C, add_dim,
        ifpenalty, rho, add_loss, update_lagmul_freq, trajs, dtype, device)
        self.dr = dr                # radius of the drone
        self.barrier = barrier
        self.num_obs = len(barrier)
        
    # try to avoid for loop in this function
    # return dim: #constraints * (num pts in all trajs) * 1
    def h(self, q):
        # avoid hitting between drones
        x = q.reshape([-1, self.dim // self.phy_dim, self.phy_dim, 1])
        x = torch.transpose(x, 0, 1)    # num of drones * num of pts * 2 * 1
        h = []
        for i in range(self.num_obs):
            h1 = torch.maximum(x[:,:,0] - self.barrier[i][0], - x[:,:,0] + self.barrier[i][1])
            h2 = torch.maximum(x[:,:,1] - self.barrier[i][2], - x[:,:,1] + self.barrier[i][3])
            h3 = torch.maximum(x[:,:,2] - self.barrier[i][4], - x[:,:,2] + self.barrier[i][5])
            h.append(torch.max(torch.stack([h1,h2,h3]), dim = 0)[0])
        h = torch.cat(h, dim = 0)
        x = q.reshape([-1, self.dim // self.phy_dim, self.phy_dim, 1])
        x = torch.transpose(x, 1, 2)
        y = torch.transpose(x, 2, 3)
        z = torch.sum((x - y)**2, dim = 1)
        mask = ~torch.eye(z.shape[1],z.shape[2], dtype=bool)
        min_value = z[:, mask].t()[...,None] - (self.dr * 2) **2
        h = torch.cat([h, min_value], dim = 0)
        return h
    
    def h_np(self, q):
        x = q.reshape([-1, self.dim // self.phy_dim, self.phy_dim, 1])
        x = np.transpose(x, (1,0,2,3))
        h = []
        for i in range(self.num_obs):
            h1 = np.maximum(x[:,:,0] - self.barrier[i][0], - x[:,:,0] + self.barrier[i][1])
            h2 = np.maximum(x[:,:,1] - self.barrier[i][2], - x[:,:,1] + self.barrier[i][3])
            h3 = np.maximum(x[:,:,2] - self.barrier[i][4], - x[:,:,2] + self.barrier[i][5])
            h.append(np.max(np.array([h1,h2,h3]), axis = 0))
        h = np.concatenate(h, axis = 0)
        # avoid hitting between drones
        x = q.reshape([-1, self.dim // self.phy_dim, self.phy_dim, 1])
        x = np.transpose(x, (0,2,1,3))
        y = np.transpose(x, (0,1,3,2))
        z = np.sum((x - y)**2, axis = 1)
        mask = ~np.eye(z.shape[1],z.shape[2], dtype=bool)
        min_value = (np.transpose(z[:, mask])[...,None]) - (self.dr * 2)**2
        h = np.concatenate([h, min_value], axis = 0)
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

def plot_anime(q_pred, net, data, filename, index_drones):
    from matplotlib import animation
    from random import randint
    from mpl_toolkits.mplot3d import Axes3D
    q_traj = q_pred.reshape([-1,100,3])
    q_diff = q_traj[1:] - q_traj[:-1]
    total_dist = np.sum(np.sqrt((q_diff ** 2).sum(-1)))
    print(total_dist)
    q_terminal = data.y_train['bd'][0,1].reshape([-1,3]).detach().cpu().numpy()
    num_drone = index_drones.size
    #num_drone = net.dim // 2
    fig = plt.figure(figsize = [8.5,5])
    ax = plt.axes(projection = '3d')
    ax.set_xlim3d([-4.0, 5.0])
    ax.set_xlabel('X')
    ax.set_ylim3d([-5.0, 5.0])
    ax.set_ylabel('Y')
    ax.set_zlim3d([0.0, 9.0])
    ax.set_zlabel('Z')
    shade = 0.4
    # block 1
    X, Y = np.meshgrid([-2, 2], [-0.5, 0.5])
    ax.plot_surface(X, Y, 7 * np.ones((2,2)) , alpha=shade, color='gray')
    ax.plot_surface(X, Y, 0 * np.ones((2, 2)), alpha=shade, color='gray')
    X, Z = np.meshgrid([-2, 2], [0, 7])
    ax.plot_surface(X, -0.5 * np.ones((2, 2)), Z, alpha=shade, color='gray')
    ax.plot_surface(X,  0.5 * np.ones((2, 2)), Z, alpha=shade, color='gray')
    Y, Z = np.meshgrid([-0.5, 0.5], [0, 7])
    ax.plot_surface(-2 * np.ones((2, 2)), Y , Z, alpha=shade, color='gray')
    ax.plot_surface( 2 * np.ones((2, 2)), Y , Z, alpha=shade, color='gray')
    # block 2
    X, Y = np.meshgrid([2, 4], [-1, 1])
    ax.plot_surface(X, Y, 4 * np.ones((2,2)) , alpha=shade, color='gray')
    ax.plot_surface(X, Y, 0 * np.ones((2, 2)), alpha=shade, color='gray')
    X, Z = np.meshgrid([2, 4], [0, 4])
    ax.plot_surface(X, -1 * np.ones((2, 2)), Z, alpha=shade, color='gray')
    ax.plot_surface(X,  1 * np.ones((2, 2)), Z, alpha=shade, color='gray')
    Y, Z = np.meshgrid([-1, 1], [0, 4])
    ax.plot_surface( 2 * np.ones((2, 2)), Y , Z, alpha=shade, color='gray')
    ax.plot_surface( 4 * np.ones((2, 2)), Y , Z, alpha=shade, color='gray')
    ax.plot(q_terminal[:,0], q_terminal[:,1], q_terminal[:,2], 'rx')
    ax.view_init(60, -30)
    colors = []
    for i in range(num_drone):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
    drones = []
    for i in range(num_drone):
        drone, = ax.plot(q_pred[:1, 3*index_drones[i]], q_pred[:1, 3*index_drones[i]+1], q_pred[:1, 3*index_drones[i]+2], c = colors[i])
        drones.append(drone)
    def init():
        for drone in drones:
            drone.set_data([],[])
        return drones
    def animate(i):
        for j in range(num_drone):
            drones[j].set_data(q_pred[:i, 3*index_drones[j]], 
                    q_pred[:i, 3*index_drones[j]+1])
            drones[j].set_3d_properties(q_pred[:i, 3*index_drones[j]+2])
        return drones
    frames = q_pred.shape[0]
    delay_time = 20000 // q_pred.shape[0]
    anim = animation.FuncAnimation(fig, animate, init_func = init,
                               frames=frames, interval=delay_time, blit=True, repeat = False)
    anim.save('figs/'+filename+'.gif', writer='imagemagick', fps=30)
    plt.show()
    plt.close()
    
# print the min value of constraints between drones, and constraints to the walls
def print_constraint(net, data, filename):
    t_grid = np.linspace(0,1,1000)[:,None]
    t_grid_tensor = torch.tensor(t_grid, dtype=net.dtype, device=net.device) # TODO: change 'cpu' to net.device
    q_pred = net.predict_q(t_grid_tensor, True)[0,...]
    num_drone = net.dim // net.phy_dim
    x = q_pred.reshape([-1, net.dim // net.phy_dim, net.phy_dim, 1])
    x = np.transpose(x, (0,2,1,3))
    y = np.transpose(x, (0,1,3,2))
    z = np.sqrt(np.sum((x - y)**2, axis = 1))
    z[:, range(num_drone), range(num_drone)] = 1
    hmin = np.amin(z)
    hmin_ind = np.unravel_index(np.argmin(z, axis=None), z.shape)[1:]
    print('min dist {}, drone radius {}, index {}'.format(hmin, net.dr, hmin_ind))
    plot_anime(q_pred, net, data, filename, np.asarray(hmin_ind))
    
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

def main():
    parser = argparse.ArgumentParser(description='Hyperparameters to be tuned')
    parser.add_argument('--l', type = float, default = 0.004, help = 'l in the soft penalty')
    parser.add_argument('--eps', type = float, default = 0.0004, help = 'epsilon in the soft penalty')
    parser.add_argument('--lam', type = float, default = 600.0, help = 'weight of boundary loss')
    parser.add_argument('--C', type = float, default = 25.0, help = 'speed limit of drones')
    parser.add_argument('--ntype', type = str, default = 'G', help = 'type of NN')
    parser.add_argument('--layers', type = int, default = 6, help = 'layers of NN')
    parser.add_argument('--width', type = int, default = 200, help = 'width of NN')
    parser.add_argument('--act', type = str, default = 'relu', help = 'activation function')
    parser.add_argument('--lagmulfreq', type = int, default = 1, help = 'the frequency of updating lagmul')
    parser.add_argument('--rho', type = float, default = 1.0, help = 'parameter for aug Lag')
    parser.add_argument('--iters', type = int, default = 0, help = 'number of iterations')
    parser.add_argument('--lbfgsiters', type = int, default = 100, help = 'number of lbfgs iterations for testcase2')
    parser.add_argument('--gpu', action="store_true", help = 'whether to use gpu')
    # modify the following flags to test
    parser.add_argument('--seed', type = int, default = 1, help = 'random seed')
    parser.add_argument('--droneradius', type = float, default = 0.2, help = 'radius of drones')
    # use this flag if loading instead of training
    parser.add_argument('--loadno', type = int, default = 300000, help = 'the no. for the model to load')
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
    train_traj = 1
    test_traj = 1
    train_noise = 0.0
    test_noise = 0.0
    traj_count = 1
    phy_dim = 3
    
    dr = args.droneradius  # radius of a drone

    q_terminal = np.array([-2.5, 2.5, 6.5,
                           -2., 2., 6.,
                           -1.5, 2.5, 6.5, 
                               -1., 2., 6.,
                               -0.5, 2.5, 6.5,
                               0., 2., 6.,
                               0.5, 2.5, 6.5,
                               1., 2., 6.,
                               1.5, 2.5, 6.5,
                               2., 2., 6.,
                               2.5, 2.5, 6.5,
                               3., 2., 6.,
                               3.5, 2.5, 6.5,
                               4., 2., 6.,
                               4.5, 2.5, 6.5,
                               -2.5, 3., 7.,
                               -2, 3.5, 7.5,
                               -1.5, 3., 7.,
                               -1, 3.5, 7.5,
                               -0.5, 3., 7.,
                               0., 3.5, 7.5,
                               0.5, 3., 7.,
                               1., 3.5, 7.5,
                               1.5, 3., 7.,
                               2., 3.5, 7.5,
                               2.5, 3., 7.,
                               3., 3.5, 7.5,
                               3.5, 3., 7.,
                               -2.5, 4.5, 8.5,
                               -2., 4., 8.,
                               -1.5, 4.5, 8.5,
                               -1., 4., 8.,
                               -0.5, 4.5, 8.5,
                               0., 4., 8.,
                               0.5, 4.5, 8.5,
                               1., 4., 8.,
                               1.5, 4.5, 8.5,
                               2., 4., 8.,
                               2.5, 4.5, 8.5,
                               3., 4., 8.,
                               3.5, 4.5, 8.5,
                               4., 4., 8.,
                               -2.5, 3.5, 5.5,
                               -2., 3., 5.,
                               -1.5, 3.5, 5.5,
                               -1., 3., 5.,
                               0, 3.5, 5.5,
                               1., 3., 5.,
                               1.5, 3.5, 5.5,
                               2., 3., 5.]).reshape([-1,3])
    q_terminal = np.concatenate((q_terminal, np.array([0, -0.5, -3]) + q_terminal), axis=0)
    q_initial = np.array([1,-1,-1]) * q_terminal + np.array([0,0,10])
    q_terminal, q_initial = q_terminal.reshape([-1]), q_initial.reshape([-1])
    print(q_terminal.shape)
    barrier = [[2, -2, 0.5, -0.5, 7, 0], [4, 2, 1, -1, 4, 0]]
    
    dim = len(q_initial)
    rho = args.rho
    l = args.l
    eps = args.eps
    lam = args.lam
    C = args.C
    ntype = args.ntype
    layers = args.layers
    width = args.width
    activation = args.act
    dtype = 'double'
    
    # training
    lr = 0.001
    iterations = args.iters
    print_every = 1000
    batch_size = None
    
    foldername = '3d_seed_{}'.format(args.seed)
    print(foldername, flush = True)
    figname = foldername

    if not os.path.exists('figs'): os.mkdir('figs')
    if not os.path.exists('figs/'+figname): os.mkdir('figs/'+figname)
    if not os.path.exists('figs/'+figname+'/best'): os.mkdir('figs/'+figname+'/best')
    if not os.path.exists('figs/'+figname+'/last'): os.mkdir('figs/'+figname+'/last')

    data = SPData(train_num, test_num, train_traj, test_traj, train_noise, test_noise, q_terminal, t_terminal, q_initial)
    if args.iters != 0:
        net = MZNN(dim, layers, width, activation, ntype, barrier, dr = dr, l = l, eps = eps, lam = lam, C = C, add_dim = 0, ifpenalty = False, 
            phy_dim = phy_dim, rho=rho, add_loss = 1, update_lagmul_freq = args.lagmulfreq, trajs = train_traj, dtype = dtype, device = device)
        callback = update_lag_mul

        args_nn = {
            'data': data,
            'net': net,
            'criterion': None,
            'optimizer': 'adam',
            'lr': lr,
            'iterations': iterations,
            'lbfgs_steps': 0,
            'path': foldername,
            'batch_size': batch_size,
            'print_every': print_every,
            'save': True,
            'callback': callback,
            'dtype': dtype,
            'device': device,
        }

        ln.Brain.Init(**args_nn)
        ln.Brain.Run()
        ln.Brain.Restore()
        ln.Brain.Output()
    else:
        net = torch.load('model/'+foldername+'/model{}.pkl'.format(args.loadno))
        data.device = device
        data.dtype = dtype
        net.device = device
        net.dtype = dtype
        
    net_plot = net

    #X_train, y_train, X_test, y_test = data.X_train, data.y_train, data.X_test, data.y_test
    #loss = np.loadtxt('outputs/'+foldername+'/loss.txt')
    #plot_cost_constraint(data, net_plot, loss, figname, print_every)

    # plot current 
    print('++++++++++++++++++++ current net +++++++++++++++++++')
    q_pred = net_plot.predict_q(data.X_test['interval'], True)[0,...]
    plot_anime(q_pred, net_plot, data, figname + '/NN', np.arange(net_plot.dim//net.phy_dim))
    print_constraint(net_plot, data, figname + '/worst_2drones')
    # print cost and hmin
    print('test cost and hmin:\n')
    compute_cost_hmin(data.X_test['interval'], net_plot, 1) # only print values for the first traj_count many trajs

    # output for best model
    print('++++++++++++++++++++ best net +++++++++++++++++++')
    net_plot = torch.load('outputs/'+foldername+'/model_best.pkl', map_location=torch.device('cpu'))
    data.device = device
    data.dtype = dtype
    net_plot.device = device
    net_plot.dtype = dtype
    q_pred = net_plot.predict_q(data.X_test['interval'], True)[0,...]
    plot_anime(q_pred, net_plot, data, figname + '/best/NN', np.arange(net_plot.dim//net.phy_dim))
    print_constraint(net_plot, data, figname + '/best/worst_2drones')
    # print cost and hmin
    print('best model: test cost and hmin:\n')
    compute_cost_hmin(data.X_test['interval'], net_plot, 1) # only print values for the first traj_count many trajs

    # output for the last one
    print('++++++++++++++++++++ last net +++++++++++++++++++')
    net_plot = torch.load('model/'+foldername+'/model{}.pkl'.format(args.iters), map_location=torch.device('cpu'))
    data.device = device
    data.dtype = dtype
    net_plot.device = device
    net_plot.dtype = dtype
    q_pred = net_plot.predict_q(data.X_test['interval'], True)[0,...]
    plot_anime(q_pred, net_plot, data, figname + '/last/NN', np.arange(net_plot.dim//net.phy_dim))
    print_constraint(net_plot, data, figname + '/last/worst_2drones')
    # print cost and hmin
    print('last model: test cost and hmin:\n')
    compute_cost_hmin(data.X_test['interval'], net_plot, 1) # only print values for the first traj_count many trajs

if __name__ == '__main__':
    main()
