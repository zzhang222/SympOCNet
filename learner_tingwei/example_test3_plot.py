#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 16:03:04 2021

@author: zzhang99
"""
import matplotlib.pyplot as plt
import torch
import os
from example_room import MZNN
import numpy as np

def plot_room_obs(q_pred, net, filename, index_drones):
    from random import randint
    num_drone = index_drones.size
    num_t = len(q_pred)
    t = [0, 1/3,2/3,1]
    t_grid = [int(tt * (num_t - 1)) for tt in t]
    colors = []
    for i in range(num_drone):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
    
    true_dr = 0.075
    for j in range(len(t)):
        fig = plt.figure(figsize = [4,4])
        ax = plt.subplot(111)
        ax.set_xlim(-net.halfroomsz-net.dr, net.halfroomsz+net.dr)
        ax.set_ylim(-net.halfroomsz-net.dr, net.halfroomsz+net.dr)
        # add obstacles 
        dr_circ = net.halfroomsz / 6.0
        center_dist = net.halfroomsz / 2.0
        circle1 = plt.Circle((-center_dist, -center_dist), dr_circ, color='k', linewidth=2, fill=False)
        circle2 = plt.Circle((-center_dist, center_dist), dr_circ, color='k', linewidth=2, fill=False)
        circle3 = plt.Circle((center_dist, -center_dist), dr_circ, color='k', linewidth=2, fill=False)
        circle4 = plt.Circle((center_dist, center_dist), dr_circ, color='k', linewidth=2, fill=False)
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        ax.add_patch(circle3)
        ax.add_patch(circle4)

        for i in range(num_drone):
            circle = plt.Circle((q_pred[t_grid[j], 2*index_drones[i]], q_pred[t_grid[j], 2*index_drones[i]+1]), true_dr, fill = False, color = colors[i])
            ax.add_patch(circle)
            ax.plot(q_pred[:t_grid[j], 2*index_drones[i]], q_pred[:t_grid[j], 2*index_drones[i]+1], c = 'gray', alpha = 0.1)
        plt.tight_layout()
        plt.savefig('figs/'+filename+str(j)+'.pdf')
        plt.show()
        plt.close()
        
def plot_room_obs_anime(q_pred, net, filename, index_drones):
    from matplotlib import animation
    from random import randint
    num_drone = index_drones.size
    fig = plt.figure(figsize = [4,4])
    xleft = np.amin(q_pred[:,::2])
    xright = np.amax(q_pred[:,::2])
    yleft = np.amin(q_pred[:,1::2])
    yright = np.amax(q_pred[:,1::2])
    ax = plt.axes(xlim=(-net.halfroomsz - net.dr, net.halfroomsz + net.dr), ylim=(-net.halfroomsz - net.dr, net.halfroomsz + net.dr))
    # add obstacles 
    dr_circ = net.halfroomsz / 6.0
    center_dist = net.halfroomsz / 2.0
    circle1 = plt.Circle((-center_dist, -center_dist), dr_circ, color='k', linewidth=2, fill=False)
    circle2 = plt.Circle((-center_dist, center_dist), dr_circ, color='k', linewidth=2, fill=False)
    circle3 = plt.Circle((center_dist, -center_dist), dr_circ, color='k', linewidth=2, fill=False)
    circle4 = plt.Circle((center_dist, center_dist), dr_circ, color='k', linewidth=2, fill=False)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)
    ax.add_patch(circle4)
    colors = []
    for i in range(num_drone):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
    drones = []
    true_dr = 0.075
    for i in range(num_drone):
        drones.append(plt.Circle((q_pred[0, 2*index_drones[i]], q_pred[0, 2*index_drones[i]+1]), true_dr, fill = False, color = colors[i]))
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

def test_room():
    foldername = 'obs4circ_room_dronesperedge32_numwall1_dr0.1_seed66_width200_layers6'
    if not os.path.exists('figs/'+foldername): os.mkdir('figs/'+foldername)
    if not os.path.exists('figs/'+foldername+'/best'): os.mkdir('figs/'+foldername+'/best')
    net_plot = torch.load('outputs/'+foldername+'/model_best.pkl')
    net_plot.device = 'cpu'
    net_plot.dtype = 'double'
    t_grid = np.linspace(0,1,1000)[:,None]
    t_grid_tensor = torch.tensor(t_grid, dtype=net_plot.dtype, device=net_plot.device)
    q_pred = net_plot.predict_q(t_grid_tensor, True)[0]
    plot_room_obs(q_pred, net_plot, foldername + '/best/NN', np.arange(net_plot.dim//2))
    plot_room_obs_anime(q_pred, net_plot, foldername + '/best/NN', np.arange(net_plot.dim//2))
 
    
if __name__ == '__main__':
    test_room()
