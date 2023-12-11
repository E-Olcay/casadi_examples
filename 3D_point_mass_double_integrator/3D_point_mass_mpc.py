# -*- coding: utf-8 -*-
"""
Main script to run an MPC to control a 3D point mass model.
The implementaiton covers the following:
- Go-To (from start to target position)
- Avoidance of one spherical obstacle 

Created on Wed Sep 21 14:22:36 2022
@author: olcay
"""

from time import time
import casadi as ca
import numpy as np
from casadi import sin, cos, pi
#from animation_code import animate, draw_control_actions
from trajectory_plot import PlotTrajectory
from Animation import TrajectoryAnimate

#---------------------------GENERAL SETTINGS ----------------------------------------------#

#MPC settings
T = 0.2     #samplig time
N = 25      #prediction horizon Nx0.2
rob_dim = 1 #we will use for avodidance from obstacles. 0.7

# state weights matrix (Q_X, Q_Y, Q_z, Q_vx, Q_vy, Q_vz)
Q_x = 1
Q_y = 1
Q_z = 1  
Q_vx = 1
Q_vy = 1
Q_vz = 1 

# controls weights matrix
R1 = 1
R2 = 1
R3 = 1

# init and target locations
x_init = 0
y_init = 0
z_init = 0
vx_init = 0
vy_init = 0
vz_init = 0

#Global target
x_target = 13
y_target = 13
z_target = 13
vx_target = 0
vy_target = 0
vz_target = 0

#control variables restriction
v_max = 1
v_min = 0

#Map Constraints
x_lb = -400
y_lb = -400
z_lb = -400
vx_lb = v_min
vy_lb = v_min
vz_lb = v_min

x_ub = 400
y_ub = 400
z_ub = 400
vx_ub = v_max
vy_ub = v_max
vz_ub = v_max

#acceleration constraints

a_x_max = 10
a_x_min = -10

a_y_max = a_x_max
a_y_min = a_x_min

a_z_max = a_x_max
a_z_min = a_x_min

#obstacle coordinates
obstackles = True  # Are there any obstackles? -> For Animation and Drawings.
obs_x = 8
obs_y =8
obs_z =8
obs_dim = 4        # Dieameter. Only circular obstacles are considered! 
n_obstacles = 1    # number of obstacles in the map
#The robot should be away from the obstacle. Sum of obs_diameter and robot diameter (rob_dim)!
#We have to consider this constraint in every prediction state. Look at the matrix g!

#Simulation Settings
sim_time = 180
save_animation = True # save animation
save_graph = True     # Save graph

#-------------------------------------------END OF GENERAL SETTINGS ---------------------------
#shift function from simulation
#shift to next step. 
def shift_timestep(T, t0, state_init, u, f): #We are not guessing!, calculating
    f_value = f(state_init, u[:, 0])         #calculate next states ?? Distrubance?
    next_state = ca.DM.full(state_init + (T * f_value))

    t0 = t0 + T
    u_init = ca.horzcat(
        u[:, 1:], #trim first entry, (because we have allready used it) 
        ca.reshape(u[:, -1], -1, 1) #repeat the last one
    )
    #in the first step, we initalized our control variable as (0,0,0)
    #after calculating the first u
    #we can initalize finally with the last u from (args['x0'] = ca.reshape(u0, n_controls*N, 1) )
    return t0, next_state, u_init

def DM2Arr(dm):
    return np.array(dm.full())

#State variables 
x=ca.SX.sym("x") #as a symbolic variable
y=ca.SX.sym("y") 
z=ca.SX.sym("z") 
vx=ca.SX.sym("vx")
vy=ca.SX.sym("vy")
vz=ca.SX.sym("vz")

states = ca.vertcat(x,y,z,vx,vy, vz)
n_states = states.numel()

#Control variables
a_x = ca.SX.sym("a_x")
a_y = ca.SX.sym("a_y")
a_z = ca.SX.sym("a_z")

controls = ca.vertcat(a_x,a_y,a_z)
n_controls = controls.numel()

rhs = ca.vertcat(vx, #state space - right hand system
       vy,
       vz,
       a_x,
       a_y,
       a_z)

f = ca.Function("f", [states,controls],[rhs]) #f(states,controls) = rhs

U = ca.SX.sym("U", n_controls, N)             #u = n_controls x N matrix
#optimization problem parameters 
P = ca.SX.sym("P",n_states + n_states) #Inıtial state and Reference State
X = ca.SX.sym("X",n_states,N+1)        #(Initial + Current)States Matrix - size 6

# state weights matrix (Q_X, Q_Y, Q_THETA)
Q = ca.diagcat(Q_x, Q_y, Q_z, Q_vx, Q_vy, Q_vz)
# controls weights matrix
R = ca.diagcat(R1, R2, R3)

g = X[:, 0] - P[:n_states]             #initial condition constraints

#MPC Contrloller (Running State Controllers)
#-----------------------------------------------------------------------------------------
#(Online Optimization) for every step
#We want too see how close we are to our target state after prediction
#How many parameters minimalize to arrive target (such as U)
#Objective Function l(x,u)=  (||x_u - x^r||^2) *Q  +  (||u-u^r||^2) *R
#Here in every step, we are looking for close we are to target 
#and how much energy (control effort) is consumed. We can penelize the terms with Q and R
#u_r = 0, u = current 
#x_r = target, at least same for N iteration (Here completly)

#Objective function is sum of all iterations, so start with 0
#-----------------------------------------------------------------------------------------

obj = 0

# #objective Function
for k in range (N):
    st = X[:,k]
    # print(i)
    con = U[:,k]
    obj = obj \
        + (st - P[n_states:]).T @ Q @ (st - P[n_states:]) \
        + con.T @ R @ con
    st_next = X[:,k+1]  #predicted value 
    f_value = f(st,con)
    st_next_euler = st + (T * f_value)         #where should be actually
    g = ca.vertcat(g, st_next - st_next_euler) #Equality constraint
    #Due to any distrubance on current state. 
    
#X and U global variables, when changing inside the loop (simulation)
#change also here. 

#__________MAIN DIFFERENCE : OBSTACLES IN EQUALITY CONSTRAINTS_________________
#We will add obstacles coinstraints to our equality coinstraints (g)
for k in range (N+1): #MPC coinstraints included st_next, thats why N+1
    g = ca.vertcat(g, ((rob_dim /2 + obs_dim/2) - (np.sqrt((X[0,k]-obs_x)**2 + (X[1,k]-obs_y)**2)))) 
    
#Find the difference between the horizons!
OPT_variables = ca.vertcat(
    X.reshape((-1, 1)),   # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
    U.reshape((-1, 1))
)

nlp_prob = {'f': obj, 'x': OPT_variables, 'g': g ,'p': P }

opts = {
    'ipopt': 
    {
        'max_iter': 2000,
        'print_level': 0,
        'acceptable_tol': 1e-8,
        'acceptable_obj_change_tol': 1e-6 #Acceptable change on objective function
    },
    'print_time': 0
}
    
    
solver = ca.nlpsol('solver', 'ipopt',nlp_prob,opts)

#For all prediction long. 3xN+1 States, 3XN for  controls
lbx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1)) #zero matrix
ubx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))

#states side
#[start:end:step]
lbx[0: n_states*(N+1): n_states] = x_lb         # x lower bound 
lbx[1: n_states*(N+1): n_states] = y_lb         # y lower bound
lbx[2: n_states*(N+1): n_states] = z_lb         # z lower bound
lbx[3: n_states*(N+1): n_states] = vx_lb        # vx lower bound
lbx[4: n_states*(N+1): n_states] = vy_lb        # vy lower bound
lbx[5: n_states*(N+1): n_states] = vz_lb        # vz lower bound

ubx[0: n_states*(N+1): n_states] = x_ub         # x upper bound
ubx[1: n_states*(N+1): n_states] = y_ub         # y upper bound
ubx[2: n_states*(N+1): n_states] = z_ub         # y upper bound
ubx[3: n_states*(N+1): n_states] = vx_ub        # vx upper bound
ubx[4: n_states*(N+1): n_states] = vy_ub        # vy upper bound
ubx[5: n_states*(N+1): n_states] = vz_ub        # vz upper bound

#Control Side
lbx[n_states*(N+1):(n_states*(N+1) + n_controls*N):n_controls] = a_x_min         # acceleration lower bound for all a
lbx[(n_states*(N+1))+1:(n_states*(N+1) + n_controls*N):n_controls] = a_y_min    
lbx[(n_states*(N+1))+2:(n_states*(N+1) + n_controls*N):n_controls] = a_z_min    

ubx[n_states*(N+1):(n_states*(N+1) + n_controls*N):n_controls] = a_x_max         # aceleration upper bound for all a
ubx[n_states*(N+1)+1: n_states*(N+1) + n_controls*N: n_controls] = a_y_max              
ubx[n_states*(N+1)+2: n_states*(N+1) + n_controls*N: n_controls] = a_z_max     

#Equality constraints g by Multiple Shooting
#All prediction long + Inequality constraints for obstackle.
lbg = ca.DM.zeros(((n_states+n_obstacles)*(N+1),1)) 
ubg = ca.DM.zeros(((n_states+n_obstacles)*(N+1),1))      

#MPC side
lbg[0: n_states*(N+1): 1] = 0
ubg[0: n_states*(N+1): 1] = 0  

#Obstacle Side, inequality constraints
lbg[n_states*(N+1): ((n_states + n_obstacles)*(N+1)): 1] = -ca.inf  #If you do not consider obstacle constraints: lbg[n_states*(N+1): ((n_states)*(N+1)): 1]
ubg[n_states*(N+1): ((n_states + n_obstacles)*(N+1)): 1] = 0        #                                             ubg[n_states*(N+1): ((n_states)*(N+1)): 1] 

#Inequality constraints due to Obstacles

args ={
       'lbx': lbx, 
       'ubx': ubx, 
       
       'lbg': lbg, # Equality constraints lower bound
       'ubg': ubg, # Equality constraints upper bound
       
       # 'p':[], #There is no parameters in this optimization problem
       # 'x0' #Initilizatioın of the optimization variable
       #           
       } 
#all of the above is just a problem setup
#--------------------------------------------------------------------------

#-------------------------------------------------------------------------
#The simulation loop should start from here

t0 = 0
state_init = ca.DM([x_init, y_init, z_init, vx_init, vy_init, vz_init])                # initial state
state_target = ca.DM([x_target, y_target, z_target, vx_target, vy_target, vz_target])  # target state

#DM is mainly used for storing matrices in CasADi
xx = ca.DM(state_init) #xx will contain the history of state
t = ca.DM(t0)

u_init = ca.DM.zeros((n_controls, N))      # initial control, three control inputs
X0 = ca.repmat(state_init, 1, N+1)         # initial state full
X0_1 = ca.reshape(X0, n_states*(N+1), 1)

#Start MPC

mpc_iter = 0 #Counter for the loop 

cat_states = DM2Arr(X0)             #Store predicted states
cat_controls = DM2Arr(u_init[:, 0]) #Store predicted contrul variables
times = np.array([[0]])

# Main Loop
# the main simulaton loop... it works as long as the error is greater
# than 10^-2 and the number of mpc steps is less than its maximum
# value.

if __name__ == '__main__':
    main_loop = time()  # return time in sec
    while (ca.norm_2(state_init - state_target) > 0.003) and (mpc_iter < sim_time):
        t1 = time()
        #For initial values, we putting inital state and target state to p
        args['p'] = ca.vertcat(  
            state_init,    # current state
            state_target   # target state
        )
        # print(args["p"][0:3])

        # optimization variable current state (for initializing)
        
        args['x0'] = ca.vertcat(
            ca.reshape(X0, n_states*(N+1), 1),
            ca.reshape(u_init, n_controls*N, 1)
        )
        #Included all predictions.

        sol = solver(
            x0=args['x0'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p=args['p']
        )
        
        #In this loop, only change first p (initial,target) then x0
        #Every time step,  the initial, current (state_init) will change
        #state target remains the same.
        #First step, we are initalizing optimization variables with zero!
        
        #Now sol have two parameters
        #sol.x -> my minimizer for the object function (Control Variables)
        #Optimal Control Variables as u
        
        u = ca.reshape(sol['x'][n_states * (N + 1):], n_controls, N)
        X0 = ca.reshape(sol['x'][: n_states * (N+1)], n_states, N+1)
        # print(u)
        # Compute Optimal solution trajectory
        # new_prediction_value =new_prediction(u,args["p"])
        # We will recieve X values for N horizon.
       
        cat_states = np.dstack(( #Stack arrays in sequence depth wise (along third axis).
            cat_states,     #old one
            DM2Arr(X0)      #new one 
        ))
        #It saved horizontally
        
        #Storing our used control value 3x1
        cat_controls = np.vstack(( #Stack arrays in sequence vertically (row wise).
            cat_controls,          #old one stored here
            DM2Arr(u[:, 0])        #new one
        ))
        #storage of all u (predicted and currently), predicted = will used next step
        #It saved vertically
        
        
        # t = np.vstack((  #save vertically all time values
        #     t,
        #     t0
        # ))
        
        #Now, next step, we applied control action;
        t0, state_init, u_init = shift_timestep(T, t0, args["p"][0:6], u, f)
        #state init = Take the next step's (now current) states
        #u_init = New initilaziton of optimization variables.
        #t0 = next time step
        
        t = np.vstack((  #save vertically all time values
            t,
            t0
        ))
        
        X0 = ca.horzcat(
            X0[:, 1:],
            ca.reshape(X0[:, -1], -1, 1)
        )
        # xx ...
        t2 = time()
        print(mpc_iter)
        print(t2-t1) #For HW implemention 
        times = np.vstack((
            times,
            t2-t1
        ))
        
        mpc_iter = mpc_iter + 1
        ss_error1 = ca.norm_2(state_init - state_target)
        print('iteration error: ', ss_error1)
        
        
    main_loop_time = time()
        
    ss_error = ca.norm_2(state_init - state_target)

    
    print('\n\n')
    print('Total time: ', main_loop_time - main_loop)
    print('avg iteration time: ', np.array(times).mean() * 1000, 'ms')
    print('final error: ', ss_error)
        
 
    #PlotTrajectory(cat_states, obs_x, obs_y, obs_z, obs_dim)
    TrajectoryAnimate(cat_states, t, np.array([x_init, y_init, z_init, x_target, y_target, z_target]), obs_x, obs_y, obs_z, obs_dim)