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
from nn_test34 import SPNN
from PS_method import PSmethod
import os
from time import perf_counter
        
class MZNN(SPNN):
    '''NN for solving the optimal control of shortest path in mazes
    '''
    def __init__(self, dim, layers, width, activation, ntype, halfroomsz = 5.0, dr = 0.2, l = 0.001, eps = 0.1, 
                 lam = 1, C = 25, add_dim = 0, rho = 1.0, add_loss = 0, 
                 update_lagmul_freq = 10, dtype = None, device = None):
        super(MZNN, self).__init__(dim, layers, width, activation, ntype, l, eps, lam, C, add_dim,
        rho, add_loss, update_lagmul_freq, dtype, device)
        self.dr = dr                # radius of the drone
        self.halfroomsz = halfroomsz  # size of half room, the range of the room is [-halfroomsz,halfroomsz]^2
    
    # try to avoid for loop in this function
    # return dim: #constraints * (num pts in all trajs) * 1
    def h(self, q):
        x = q.reshape([-1, self.dim // 2, 2, 1])
        x = torch.transpose(x, 0, 1)    # num of drones * num of pts * 2 * 1
        h1 = x[:,:,0] + self.halfroomsz             # x >= -C
        h2 = - x[:,:,0] + self.halfroomsz           # x <= C
        h3 = x[:,:,1] + self.halfroomsz             # y >= -C
        h4 = -x[:,:,1] + self.halfroomsz            # y <= C
        h = torch.cat([h1,h2,h3,h4], dim = 0)
        # avoid hitting between drones
        x = q.reshape([-1, self.dim // 2, 2, 1])
        x = torch.transpose(x, 1, 2)
        y = torch.transpose(x, 2, 3)
        z = torch.sum((x - y)**2, dim = 1)
        mask = ~torch.eye(z.shape[1],z.shape[2], dtype=bool)
        min_value = z[:, mask].t()[...,None] - (self.dr * 2) **2
        h = torch.cat([h, min_value], dim = 0)
        return h
    
    def h_np(self, q):
        x = q.reshape([-1, self.dim // 2, 2, 1])
        x = np.transpose(x, (1,0,2,3))
        h1 = x[:,:,0] + self.halfroomsz             # x >= -C
        h2 = - x[:,:,0] + self.halfroomsz           # x <= C
        h3 = x[:,:,1] + self.halfroomsz             # y >= -C
        h4 = -x[:,:,1] + self.halfroomsz            # y <= C
        h = np.concatenate([h1, h2, h3, h4], axis = 0)
        # avoid hitting between drones
        x = q.reshape([-1, self.dim // 2, 2, 1])
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
    cost, hmin = compute_cost_hmin(data.X_train['interval'], net)
    return cost, hmin
           
def compute_cost_hmin(t, net):
    cost = torch.mean(net.value_function(t)).item()
    hmin = net.hmin_function(t).item()
    print('cost value: {}\t'.format(cost))
    print('constraint value: {}\n'.format(hmin), flush=True)
    return cost, hmin


def plot_anime(q_pred, net, filename, index_drones):
    from matplotlib import animation
    from random import randint
    num_drone = index_drones.size
    fig = plt.figure(figsize = [4,4])
    xleft = np.amin(q_pred[:,::2])
    xright = np.amax(q_pred[:,::2])
    yleft = np.amin(q_pred[:,1::2])
    yright = np.amax(q_pred[:,1::2])
    ax = plt.axes(xlim=(-net.halfroomsz - net.dr, net.halfroomsz + net.dr), ylim=(-net.halfroomsz - net.dr, net.halfroomsz + net.dr))
    colors = []
    for i in range(num_drone):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
    drones = []
    for i in range(num_drone):
        drones.append(plt.Circle((q_pred[0, 2*index_drones[i]], q_pred[0, 2*index_drones[i]+1]), net.dr, fill = False, color = colors[i]))
    def init():
        for i in range(num_drone):
            ax.add_patch(drones[i])
        return drones 
    def animate(i):
        for j in range(num_drone):
            drones[j].center = (q_pred[i,2*index_drones[j]], q_pred[i,2*index_drones[j]+1])
        return drones
    frames = q_pred.shape[0]
    delay_time = 20000 // q_pred.shape[0]
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=frames, interval=delay_time, blit=True, repeat = False)
    anim.save('figs/'+filename+'.gif', writer='imagemagick', fps=30)
    plt.show()
    plt.close()
    
# print the min value of constraints between drones, and constraints to the walls
def print_constraint(net, data, filename):
    t_grid = np.linspace(0,1,1000)[:,None]
    t_grid_tensor = torch.tensor(t_grid, dtype=net.dtype, device=net.device) # TODO: change 'cpu' to net.device
    q_pred = net.predict_q(t_grid_tensor, True)[0,...]
    num_drone = net.dim // 2
    x = q_pred.reshape([-1, net.dim // 2, 2, 1])
    x = np.transpose(x, (0,2,1,3))
    y = np.transpose(x, (0,1,3,2))
    z = np.sqrt(np.sum((x - y)**2, axis = 1))
    z[:, range(num_drone), range(num_drone)] = 1000
    hmin = np.amin(z)
    hmin_ind = np.unravel_index(np.argmin(z, axis=None), z.shape)[1:]
    print('min dist {}, drone radius {}, index {}'.format(hmin, net.dr, hmin_ind))
    plot_anime(q_pred, net, filename, np.asarray(hmin_ind))
    
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
    parser.add_argument('--width', type = int, default = 60, help = 'width of NN')
    parser.add_argument('--act', type = str, default = 'relu', help = 'activation function')
    parser.add_argument('--lagmulfreq', type = int, default = 1, help = 'the frequency of updating lagmul')
    parser.add_argument('--rho', type = float, default = 1.0, help = 'parameter for aug Lag')
    parser.add_argument('--iters', type = int, default = 0, help = 'number of iterations')
    parser.add_argument('--gpu', action="store_true", help = 'whether to use gpu')
    # modify the following flags to test
    parser.add_argument('--seed', type = int, default = 0, help = 'random seed')
    parser.add_argument('--numdperedge', type = int, default = 19, help = 'number of drones per edge, the total number is 4 times this number')
    parser.add_argument('--halfroomsz', type = float, default = 5.0, help = 'size of half room')
    parser.add_argument('--droneradius', type = float, default = 0.15, help = 'radius of drones')
    parser.add_argument('--numwall', type=int, default = 4, help = 'number of vehicle walls')
    # use this flag if loading instead of training
    parser.add_argument('--loadno', type = int, default = 0, help = 'the no. for the model to load')
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
    
    halfroomsz = args.halfroomsz
    dr = args.droneradius  # radius of a drone

    gridpt = np.linspace(-halfroomsz, halfroomsz, args.numdperedge + 1)
    q_initial = []
    for i in range(args.numwall):
        drones_left = np.concatenate([gridpt[i]*np.ones([args.numdperedge-1-2*i,1]), gridpt[1+i:-1-i,None]], axis = -1)  # don't contain end pts
        drones_upper = np.concatenate([gridpt[i:args.numdperedge+1-i,None], gridpt[-1-i]*np.ones([args.numdperedge+1-2*i,1])], axis = -1)  # contains both end pts
        drones_left = drones_left.reshape([-1])
        drones_upper = drones_upper.reshape([-1])
        q_initial.extend([drones_left, drones_upper, -drones_left, -drones_upper])
    q_initial = np.concatenate(q_initial, axis = 0)
    q_terminal = -q_initial

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

    print('dim {}'.format(dim))
    print('grid point dist {}, dr {}'.format(gridpt[1]-gridpt[0], args.droneradius))
    
    # training
    lr = 0.001
    iterations = args.iters
    print_every = 1000
    batch_size = None
    
    foldername = ''
    foldername += 'room_dronesperedge{}'.format(args.numdperedge)
    foldername += '_halfroomsz{}'.format(args.halfroomsz)
    foldername += '_numwall{}'.format(args.numwall)
    foldername += '_dr{}'.format(args.droneradius)
    foldername += '_seed{}'.format(args.seed)
    foldername += '_width{}'.format(args.width)
    foldername += '_layers{}'.format(args.layers)
    foldername += '_iters{}'.format(args.iters)
    print(foldername, flush = True)
    figname = foldername

    if not os.path.exists('figs'): os.mkdir('figs')
    if not os.path.exists('figs/'+figname): os.mkdir('figs/'+figname)
    if not os.path.exists('figs/'+figname+'/best'): os.mkdir('figs/'+figname+'/best')
    if not os.path.exists('figs/'+figname+'/last'): os.mkdir('figs/'+figname+'/last')

    data = SPData(train_num, test_num, train_traj, test_traj, train_noise, test_noise, q_terminal, t_terminal, q_initial)
    if args.loadno == 0:
        net = MZNN(dim, layers, width, activation, ntype, halfroomsz = halfroomsz, dr = dr, l = l, eps = eps, lam = lam, C = C, add_dim = 0, 
            rho=rho, add_loss = 1, update_lagmul_freq = args.lagmulfreq, dtype = dtype, device = device)
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
        net = torch.load('model/'+foldername+'/model{}.pkl'.format(args.loadno), map_location=torch.device('cpu'))
        data.device = device
        data.dtype = dtype
        net.device = device
        net.dtype = dtype

    net_plot = net

    # plot current 
    print('++++++++++++++++++++ current net +++++++++++++++++++')
    q_pred = net_plot.predict_q(data.X_test['interval'], True)[0,...]
    plot_anime(q_pred, net_plot, figname + '/NN', np.arange(net_plot.dim//2))
    print_constraint(net_plot, data, figname + '/worst_2drones')
    # print cost and hmin
    print('test cost and hmin:\n')
    compute_cost_hmin(data.X_test['interval'], net_plot) 

    # output for best model
    print('++++++++++++++++++++ best net +++++++++++++++++++')
    net_plot = torch.load('outputs/'+foldername+'/model_best.pkl', map_location=torch.device('cpu'))
    data.device = device
    data.dtype = dtype
    net_plot.device = device
    net_plot.dtype = dtype
    q_pred = net_plot.predict_q(data.X_test['interval'], True)[0,...]
    plot_anime(q_pred, net_plot, figname + '/best/NN', np.arange(net_plot.dim//2))
    print_constraint(net_plot, data, figname + '/best/worst_2drones')
    # print cost and hmin
    print('best model: test cost and hmin:\n')
    compute_cost_hmin(data.X_test['interval'], net_plot)
    
    
if __name__ == '__main__':
    main()
