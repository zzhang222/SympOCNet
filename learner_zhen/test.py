#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 11:09:53 2021

@author: zzhang99
"""

from mpl_toolkits.mplot3d import art3d
import matplotlib.pyplot as plt
import numpy as np

def rotation_matrix(d):
    """
    Calculates a rotation matrix given a vector d. The direction of d
    corresponds to the rotation axis. The length of d corresponds to 
    the sin of the angle of rotation.

    Variant of: http://mail.scipy.org/pipermail/numpy-discussion/2009-March/040806.html
    """
    sin_angle = np.linalg.norm(d)

    if sin_angle == 0:
        return np.identity(3)

    d /= sin_angle

    eye = np.eye(3)
    ddt = np.outer(d, d)
    skew = np.array([[    0,  d[2],  -d[1]],
                  [-d[2],     0,  d[0]],
                  [d[1], -d[0],    0]], dtype=np.float64)

    M = ddt + np.sqrt(1 - sin_angle**2) * (eye - ddt) + sin_angle * skew
    return M

def pathpatch_2d_to_3d(pathpatch, z = 0, normal = 'z'):
    """
    Transforms a 2D Patch to a 3D patch using the given normal vector.

    The patch is projected into they XY plane, rotated about the origin
    and finally translated by z.
    """
    if type(normal) is str: #Translate strings to normal vectors
        index = "xyz".index(normal)
        normal = np.roll((1.0,0,0), index)

    normal /= np.linalg.norm(normal) #Make sure the vector is normalised

    path = pathpatch.get_path() #Get the path and the associated transform
    trans = pathpatch.get_patch_transform()

    path = trans.transform_path(path) #Apply the transform

    pathpatch.__class__ = art3d.PathPatch3D #Change the class
    pathpatch._code3d = path.codes #Copy the codes
    pathpatch._facecolor3d = pathpatch.get_facecolor #Get the face color    

    verts = path.vertices #Get the vertices in 2D

    d = np.cross(normal, (0, 0, 1)) #Obtain the rotation vector    
    M = rotation_matrix(d) #Get the rotation matrix

    pathpatch._segment3d = np.array([np.dot(M, (x, y, 0)) + (0, 0, z) for x, y in verts])

def pathpatch_translate(pathpatch, delta):
    """
    Translates the 3D pathpatch by the amount delta.
    """
    pathpatch._segment3d += delta
    
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import Circle
from itertools import product
from matplotlib import animation
    
fig = plt.figure(figsize = [4,4])
ax = plt.axes(projection = '3d')
drones = []
for i in range(1):
    drones.append(plt.Circle((0, 0), .5, fill = False, color = 'y'))
def init():
    for i in range(1):
        ax.add_patch(drones[i])
        pathpatch_2d_to_3d(drones[i], z = 0, normal = [1,-1,1])
    return drones 
def animate(i):
    for j in range(1):
        #drones[j].center = (q_pred[i,2*index_drones[j]], q_pred[i,2*index_drones[j]+1])
        pathpatch_translate(drones[j], (0.005, 0.01, 0))
    return drones
frames = 200
delay_time = 20000 // 200
anim = animation.FuncAnimation(fig, animate, init_func=init,
                           frames=frames, interval=delay_time, blit=True, repeat = False)
anim.save('figs/test.gif', writer='imagemagick', fps=30)
plt.show()
#plt.close()