# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 10:53:21 2022

@author: olcay
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np



def PlotTrajectory(cat_states, obs_x, obs_y, obs_z, obs_dim):
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(2,3,4) # plot the point (2,3,4) on the figure
    
    
    #Draw sphere for obstacles   
    def plt_sphere(obs_x, obs_y, obs_z, obs_dim):            
        # a complete sphere
        R = obs_dim/2
        theta = np.linspace(0, 2 * np.pi, 1000)
        phi = np.linspace(0, np.pi, 1000)
        x_sphere = R * np.outer(np.cos(theta), np.sin(phi)) + obs_x
        y_sphere = R * np.outer(np.sin(theta), np.sin(phi)) + obs_y
        z_sphere = R * np.outer(np.ones(np.size(theta)), np.cos(phi)) + obs_z
        
        # a complete circle on the sphere
        x_circle = R * np.sin(theta)
        y_circle = R * np.cos(theta)
        
        # 3d plot
        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x_sphere, y_sphere, z_sphere, color='blue', alpha=0.5)
        #ax.plot(x_circle, y_circle, 0, color='green')
        
        
    # get variables
    x = cat_states[0, 0, :]
    y = cat_states[1, 0, :]
    z = cat_states[2, 0, :]
    
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x,y,z, color='red') 
    
    plt_sphere(obs_x, obs_y, obs_z, obs_dim) 
    plt.show()   