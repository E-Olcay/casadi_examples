# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 08:55:00 2022

@author: olcay
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 17:02:42 2022

@author: olcay
"""

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation


def TrajectoryAnimate(cat_states, t, reference, obs_x, obs_y, obs_z, obs_dim):
    
    
    # Attaching 3D axis to the figure
    fig = plt.figure(figsize=(8,8), dpi=(1920/16))
    ax = p3.Axes3D(fig)    
      
    # Setting the axes properties
    ax.set_xlim3d([0.0, 15.0])
    ax.set_xlabel('X')
    
    ax.set_ylim3d([0.0, 15.0])
    ax.set_ylabel('Y')
    
    ax.set_zlim3d([0.0, 15.0])
    ax.set_zlabel('Z')

    ax.set_xlabel('X-axis [m]')
    ax.set_ylabel('Y-axis [m]')
    ax.set_zlabel('Z-axis [m]')   
    
    time_display = ax.text(22.0, 1.0, 39.0, "red" ,color='red', transform=ax.transAxes)
    
    
    current_pos, = ax.plot([0,0], [0,0], [0,0], 'b.')
    horizon, = ax.plot([0,0], [0,0], [0,0], 'b.')
    
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


    #def init():
    #    return path
    
    def update(i):
        global current_pos, horizon
        current_pos.remove()
        horizon.remove()
        
        # get variables
        x = cat_states[0, 0, i]
        y = cat_states[1, 0, i]
        z = cat_states[2, 0, i]

        x_hor = cat_states[0, :, i]
        y_hor = cat_states[1, :, i]
        z_hor = cat_states[2, :, i]        
        
       
        # current pos
        current_pos, = ax.plot(x, y, z, "ro", markersize=7)
        
        # prediction horizon
        horizon, = ax.plot(x_hor, y_hor, z_hor, "gx", markersize=5)
       
       #path, = ax.plot([], [], [], 'k', linewidth=2) 
       #path.set_data([x],[y])
       #path.set_3d_properties([z])
        
        # path
        ax.plot(x, y, z, "k.", markersize=2)
        
        time_display.set_text('Simulation time = %.1fs' % (t[i]))
        
    
    plt_sphere(obs_x, obs_y, obs_z, obs_dim)
    ax.scatter(reference[3], reference[4], reference[5]) # plot the target position on the figure
    
    # Creating the Animation object
    anim = animation.FuncAnimation(fig, update, interval = 100*0.2) #63
    
    anim.save('./animation_3D_mpc'+'.gif', writer='ffmpeg', fps=10)
    plt.show()