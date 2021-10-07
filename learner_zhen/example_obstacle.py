#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 12:21:59 2021

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
        
class MZNN(SPNN):
    '''NN for solving the optimal control of shortest path in mazes
    '''
    def __init__(self, dim, layers, width, activation, ntype, halfroomsz = 5.0, dr = 0.2, ws = 0, qr = 0, l = 0.001, eps = 0.1, 
                 phy_dim = 2, lam = 1, C = 25, add_dim = 0, ifpenalty = True, rho = 1.0, add_loss = 0, 
                 update_lagmul_freq = 10, trajs = 1, dtype = None, device = None):
        super(MZNN, self).__init__(dim, phy_dim, layers, width, activation, ntype, l, eps, lam, C, add_dim,
        ifpenalty, rho, add_loss, update_lagmul_freq, trajs, dtype, device)
        self.dr = dr                # radius of the drone
        self.ws = ws                # centre the obstacles
        self.num_obs = len(ws)      
        self.qr = qr                # width of the obstacles
        self.halfroomsz = halfroomsz # size of half room, the range of the room is [-halfroomsz,halfroomsz]^2
    
    # try to avoid for loop in this function
    # return dim: #constraints * (num pts in all trajs) * 1
    def h(self, q):
        ws = torch.tensor(self.ws, device = self.device, dtype = self.dtype)
        h_obs = torch.sum((q.reshape([-1,1,2]) - ws[None,...])**2, dim = -1, keepdims = True) - (self.dr + self.qr) ** 2
        h_obs = h_obs.reshape([-1, self.dim // 2 * self.num_obs, 1])
        h_obs = torch.transpose(h_obs, 0, 1)
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
        h = torch.cat([h, h_obs, min_value], dim = 0)
        return h
    
    def h_np(self, q):
        h_obs = np.sum((q.reshape([-1,1,2]) - ws[None,...])**2, axis = -1, keepdims = True) - (self.dr + self.qr) ** 2
        h_obs = h_obs.reshape([-1, self.dim // 2 * self.num_obs, 1])
        h_obs = np.transpose(h_obs, (1,0,2))
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
        h = np.concatenate([h, h_obs, min_value], axis = 0)
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

def plot_anime(q_pred, net, filename, index_drones):
    from matplotlib import animation
    from random import randint
    num_drone = index_drones.size
    x = np.linspace(-net.halfroomsz, net.halfroomsz, 100)
    y = np.linspace(-net.halfroomsz, net.halfroomsz, 100)
    xx, yy = np.meshgrid(x, y)
    zz = []
    for i, w in enumerate(net.ws):
        w = np.array([w])
        p = np.concatenate([xx.reshape([-1,1]), yy.reshape([-1,1])], axis = -1)
        h = ((w - p) ** 2).sum(-1) - net.qr ** 2
        zz.append(h.reshape([100,100]))
    zz = np.min(np.array(zz), axis = 0)
    fig = plt.figure(figsize = [4,4])
    ax = plt.axes(xlim=(-net.halfroomsz, net.halfroomsz), ylim=(-net.halfroomsz, net.halfroomsz))
    plt.contour(xx,yy,zz,[0])
    colors = []
    for i in range(num_drone):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
    drones = []
    for i in range(num_drone):
        drones.append(plt.Circle((q_pred[0, 2*index_drones[i]], q_pred[0, 2*i+1]), net.dr, fill = False, color = colors[i]))
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
    
def print_constraint(net, data, filename):
    t_grid = np.linspace(0,1,1000)[:,None]
    t_grid_tensor = torch.tensor(t_grid, dtype=net.dtype, device=net.device) # TODO: change 'cpu' to net.device
    q_pred = net.predict_q(t_grid_tensor, True)[0,...]
    num_drone = net.dim // 2
    x = q_pred.reshape([-1, net.dim // 2, 2, 1])
    x = np.transpose(x, (0,2,1,3))
    y = np.transpose(x, (0,1,3,2))
    z = np.sqrt(np.sum((x - y)**2, axis = 1))
    z[:, range(num_drone), range(num_drone)] = 1
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
    parser.add_argument('--ntype', type = str, default = 'G', help = 'type of NN')
    parser.add_argument('--layers', type = int, default = 6, help = 'layers of NN')
    parser.add_argument('--width', type = int, default = 60, help = 'width of NN')
    parser.add_argument('--act', type = str, default = 'relu', help = 'activation function')
    parser.add_argument('--lagmulfreq', type = int, default = 1, help = 'the frequency of updating lagmul')
    parser.add_argument('--rho', type = float, default = 1.0, help = 'parameter for aug Lag')
    parser.add_argument('--iters', type = int, default = 100000, help = 'number of iterations')
    parser.add_argument('--lbfgsiters', type = int, default = 100, help = 'number of lbfgs iterations for testcase2')
    parser.add_argument('--gpu', action="store_true", help = 'whether to use gpu')
    # modify the following flags to test
    parser.add_argument('--seed', type = int, default = 1, help = 'random seed')
    parser.add_argument('--numdperedge', type = int, default = 32, help = 'number of drones per edge, the total number is 4 times this number')
    parser.add_argument('--halfroomsz', type = float, default = 5.0, help = 'size of half room')
    parser.add_argument('--droneradius', type = float, default = 0.1, help = 'radius of drones')
    # use this flag if loading instead of training
    parser.add_argument('--loadno', type = int, default = 0, help = 'the no. for the model to load')
    parser.add_argument('--rowobs', type = int, default = 3, help = 'number of obstacle per row, the total number is the square of this number')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.gpu:
        device = 'gpu' # 'cpu' or 'gpu'
    else:
        device = 'cpu' # 'cpu' or 'gpu'

    device = 'gpu'
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
    drones_left = np.concatenate([np.zeros([args.numdperedge - 1, 1]) - halfroomsz, gridpt[1:-1,None]], axis = -1)  # don't contain end pts
    drones_upper = np.concatenate([gridpt[:,None], np.zeros([args.numdperedge + 1, 1]) + halfroomsz], axis = -1)  # contains both end pts
    drones_left = drones_left.reshape([-1])
    drones_upper = drones_upper.reshape([-1])
    q_initial = np.concatenate([drones_left, drones_upper, -drones_left, -drones_upper], axis = 0)
    q_terminal = -q_initial
    
    k = args.rowobs
    qr = 0.3 * halfroomsz / k
    x = np.linspace(-halfroomsz/2, halfroomsz/2, k)
    y = np.linspace(-halfroomsz/2, halfroomsz/2, k)
    xx, yy = np.meshgrid(x, y)
    ws = np.concatenate([xx.reshape([-1,1]), yy.reshape([-1,1])], axis = -1)

    dim = len(q_initial)
    rho = args.rho
    l = args.l
    eps = args.eps
    lam = args.lam
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
    
    foldername = 'room_droneperedge_{}_obsperrow_{}'.format(args.numdperedge, k)
    print(foldername, flush = True)
    figname = foldername

    if not os.path.exists('figs'): os.mkdir('figs')
    if not os.path.exists('figs/'+figname): os.mkdir('figs/'+figname)

    data = SPData(train_num, test_num, train_traj, test_traj, train_noise, test_noise, q_terminal, t_terminal, q_initial)
    if args.loadno == 0:
        net = MZNN(dim, layers, width, activation, ntype, halfroomsz = halfroomsz, dr = dr, ws = ws, qr = qr, l = l, eps = eps, lam = lam, add_dim = 0, ifpenalty = False, 
            rho=rho, add_loss = 1, update_lagmul_freq = args.lagmulfreq, trajs = train_traj, dtype = dtype, device = device)
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
    plot_anime(q_pred, net_plot, figname + '/NN', np.arange(net_plot.dim//2))
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
    plot_anime(q_pred, net_plot, figname + '/best/NN', np.arange(net_plot.dim//2))
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
    plot_anime(q_pred, net_plot, figname + '/last/NN', np.arange(net_plot.dim//2))
    print_constraint(net_plot, data, figname + '/last/worst_2drones')
    # print cost and hmin
    print('last model: test cost and hmin:\n')
    compute_cost_hmin(data.X_test['interval'], net_plot, 1) # only print values for the first traj_count many trajs

if __name__ == '__main__':
    main()