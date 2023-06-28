# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 10:47:06 2022

@author: olcay
"""

import matplotlib.pyplot as plt



def draw_trajectory(cat_states):
    
    x_1 = cat_states[0, 0, :] #in 2nd colmn is horizon-> here current state
    y_1 = cat_states[1, 0, :]
    
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set(xlabel='x', ylabel='y')
    ax1.plot(x_1,y_1)
    fig.suptitle('Position')
    plt.grid()
    plt.show()

def draw_control_actions(cat_controls,t):
    #more resolution
    plot_controls_u1 = cat_controls[::2]
    plot_controls_u2 = cat_controls[1:(len(cat_controls)):2]
    plot_time = t
    
    
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax1.set(xlabel='Time (t)', ylabel='u1 (m/s^2)')
    ax1.step(plot_time,plot_controls_u1,'tab:orange')
    plt.grid()
    
    
    ax2 = fig.add_subplot(2,1,2)
    ax2.set(xlabel='Time (t)', ylabel='u2 (rad)')
    ax2.step(plot_time,plot_controls_u2,'tab:green')

    
    fig.suptitle('Control Actions (u1, u2)')
    plt.grid()
    fig.show()

    
    
