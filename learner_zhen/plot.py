#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 16:03:04 2021

@author: zzhang99
"""
import matplotlib.pyplot as plt
import torch
from example_room import MZNN
import numpy as np

def plot_room(q_pred, net, filename, index_drones):
    from random import randint
    num_drone = index_drones.size
    num_t = len(q_pred)
    t = [0,1/3,2/3,1]
    t_grid = [int(tt * (num_t - 1)) for tt in t]
    colors = []
    for i in range(num_drone):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
    
    #num_drone = net.dim // 2
    for j in range(len(t)):
        fig = plt.figure(figsize = [4,4])
        ax = plt.subplot(111)
        ax.set_xlim(-net.halfroomsz-net.dr, net.halfroomsz+net.dr)
        ax.set_ylim(-net.halfroomsz-net.dr, net.halfroomsz+net.dr)
        for i in range(num_drone):
            circle = plt.Circle((q_pred[t_grid[j], 2*index_drones[i]], q_pred[t_grid[j], 2*index_drones[i]+1]), net.dr*9/10, fill = False, color = colors[i])
            ax.add_patch(circle)
            ax.plot(q_pred[:t_grid[j], 2*index_drones[i]], q_pred[:t_grid[j], 2*index_drones[i]+1], c = 'gray', alpha = 0.1)
        plt.tight_layout()
        plt.savefig('figs/'+filename+str(j)+'.pdf')
        plt.show()
        plt.close()
        
def plot_swarm(q_pred, net, q_terminal, filename, index_drones):
    from random import randint
    from mpl_toolkits.mplot3d import Axes3D
    num_drone = index_drones.size
    num_t = len(q_pred)
    t = [1/3,2/3,1]
    t_grid = [int(tt * (num_t - 1)) for tt in t]
    colors = []
    for i in range(num_drone):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
    #num_drone = net.dim // 2
    for j in range(3):
        fig = plt.figure(figsize = [5,5])
        ax = plt.axes(projection = '3d')
        ax.set_xlim3d([-4.0, 5.0])
        ax.set_xlabel('X')
        ax.set_ylim3d([-5.0, 5.0])
        ax.set_ylabel('Y')
        ax.set_zlim3d([0.0, 9.0])
        ax.set_zlabel('Z')
        shade = 0.4
        # block 1
        X, Y = np.meshgrid([-1.8, 1.8], [-0.3, 0.3])
        ax.plot_surface(X, Y, 7 * np.ones((2,2)) , alpha=shade, color='gray')
        ax.plot_surface(X, Y, 0 * np.ones((2, 2)), alpha=shade, color='gray')
        X, Z = np.meshgrid([-1.8, 1.8], [0.2, 6.8])
        ax.plot_surface(X, -0.5 * np.ones((2, 2)), Z, alpha=shade, color='gray')
        ax.plot_surface(X,  0.5 * np.ones((2, 2)), Z, alpha=shade, color='gray')
        Y, Z = np.meshgrid([-0.3, 0.3], [0.2, 6.8])
        ax.plot_surface(-2 * np.ones((2, 2)), Y , Z, alpha=shade, color='gray')
        ax.plot_surface( 2 * np.ones((2, 2)), Y , Z, alpha=shade, color='gray')
        # block 2
        X, Y = np.meshgrid([2.2, 3.8], [-0.8, 0.8])
        ax.plot_surface(X, Y, 4 * np.ones((2,2)) , alpha=shade, color='gray')
        ax.plot_surface(X, Y, 0 * np.ones((2, 2)), alpha=shade, color='gray')
        X, Z = np.meshgrid([2.2, 3.8], [0.2, 3.8])
        ax.plot_surface(X, -1 * np.ones((2, 2)), Z, alpha=shade, color='gray')
        ax.plot_surface(X,  1 * np.ones((2, 2)), Z, alpha=shade, color='gray')
        Y, Z = np.meshgrid([-0.8, 0.8], [0.2, 3.8])
        ax.plot_surface( 2 * np.ones((2, 2)), Y , Z, alpha=shade, color='gray')
        ax.plot_surface( 4 * np.ones((2, 2)), Y , Z, alpha=shade, color='gray')
        ax.plot(q_terminal[:,0], q_terminal[:,1], q_terminal[:,2], 'rx')
        ax.view_init(60, -20)
        for i in range(num_drone):
            ax.plot(q_pred[:t_grid[j], 3*index_drones[i]], q_pred[:t_grid[j], 3*index_drones[i]+1], q_pred[:t_grid[j], 3*index_drones[i]+2], c = colors[i])
        plt.tight_layout()
        plt.savefig('figs/'+filename+str(j)+'.pdf')
        plt.show()
        plt.close()
    
def test_room():
    foldername = 'room_dronesperedge_19_seed_0'
    net_plot = torch.load('outputs/'+foldername+'/model_best.pkl')
    net_plot.device = 'cpu'
    net_plot.dtype = 'double'
    t_grid = np.linspace(0,1,1000)[:,None]
    t_grid_tensor = torch.tensor(t_grid, dtype=net_plot.dtype, device=net_plot.device)
    q_pred = net_plot.predict_q(t_grid_tensor, True)[0]
    plot_room(q_pred, net_plot, foldername + '/best/room_256_NN', np.arange(net_plot.dim//2))
 
def test_swarm():
    foldername = '3d_seed_1'
    net_plot = torch.load('outputs/'+foldername+'/model_best.pkl')
    net_plot.device = 'cpu'
    net_plot.dtype = 'double'
    y_train = np.load('outputs/'+foldername+'/y_train.npz')
    q_terminal = y_train['bd'][0,1].reshape([-1,3])
    t_grid = np.linspace(0,1,1000)[:,None]
    t_grid_tensor = torch.tensor(t_grid, dtype=net_plot.dtype, device=net_plot.device)
    q_pred = net_plot.predict_q(t_grid_tensor, True)[0]
    plot_swarm(q_pred, net_plot, q_terminal, foldername + '/best/3d_NN', np.arange(net_plot.dim//3))
    
if __name__ == '__main__':
    test_room()