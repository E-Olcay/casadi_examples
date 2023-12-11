# -*- coding: utf-8 -*-
"""
Main script to run an MPC to control a 3D point mass (single-integrator) model.
Implementaiton covers the following:
- Go-To (from start to target position)
- Avoidance of one predefined spherical obstacles 

Created on Wed Sep 21 14:22:36 2022
@author: olcay
"""

from time import time
import casadi as ca
import numpy as np
from Animation import TrajectoryAnimate

#---------------------------GENERAL SETTINGS ----------------------------------------------#

#MPC settings
T = 0.2 #samplig time
N = 150  #prediction horizon Nx0.2
rob_dim = 1 # for avodidance of obstacles.

# state weights matrix (Q_X, Q_Y, Q_z)
Q_x = 1
Q_y = 1
Q_z = 1  


# controls weights matrix
R1 = 1
R2 = 1
R3 = 1

# init and target locations
x_init = 0
y_init = 0
z_init = 0

#Global target
x_target = 13
y_target = 13
z_target = 13

#control variables restriction
v_max = 2
v_min = 0

#Map Constraints
x_lb = -400
y_lb = -400
z_lb = -400


x_ub = 400
y_ub = 400
z_ub = 400


#obstacle coordinates
obstacles = True   # Are there any obstacles? -> For Animation and Drawings.
n_obstacles = 1    # number of obstacles in the map
obs_x = 8
obs_y = 8
obs_z = 8
obs_dim = 4        # Dieameter. Only circular obstacles are considered! 
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


states = ca.vertcat(x,y,z)
n_states = states.numel()

#Control variables
v_x = ca.SX.sym("v_x")
v_y = ca.SX.sym("v_y")
v_z = ca.SX.sym("v_z")

controls = ca.vertcat(v_x,v_y,v_z)
n_controls = controls.numel()

rhs = ca.vertcat(v_x, #state space - right hand system
       v_y,
       v_z
       )

f = ca.Function("f", [states,controls],[rhs]) #f(states,controls) = rhs

U = ca.SX.sym("U", n_controls, N)             #u = n_controls x N matrix
#optimization problem parameters 
P = ca.SX.sym("P",n_states + n_states) #Inıtial state and Reference State
X = ca.SX.sym("X",n_states,N+1)        #(Initial + Current)States Matrix - size 3

# state weights matrix (Q_X, Q_Y, Q_THETA)
Q = ca.diagcat(Q_x, Q_y, Q_z)
# controls weights matrix
R = ca.diagcat(R1, R2, R3)

g = X[:, 0] - P[:n_states]             #initial condition constraints

#MPC Contrloller (Running State Controllers)
#-----------------------------------------------------------------------------------------
#(Online Optimization) in every time step
#We want to see how close we come to our target state after prediction
#Objective Function l(x,u)=  (||x_u - x^r||^2) *Q  +  (||u-u^r||^2) *R
#u_r = 0, u = current 
#x_r = target, same for N iteration 

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
#We add obstacle coinstraints to our equality coinstraints (g)

if obstacles == True:
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

ubx[0: n_states*(N+1): n_states] = x_ub         # x upper bound
ubx[1: n_states*(N+1): n_states] = y_ub         # y upper bound
ubx[2: n_states*(N+1): n_states] = z_ub         # y upper bound

#Control Side
lbx[n_states*(N+1):(n_states*(N+1) + n_controls*N):n_controls] = v_min         # velocity lower bound for all v
lbx[(n_states*(N+1))+1:(n_states*(N+1) + n_controls*N):n_controls] = v_min    
lbx[(n_states*(N+1))+2:(n_states*(N+1) + n_controls*N):n_controls] = v_min    

ubx[n_states*(N+1):(n_states*(N+1) + n_controls*N):n_controls] = v_max         # velocity upper bound for all v
ubx[n_states*(N+1)+1: n_states*(N+1) + n_controls*N: n_controls] = v_max              
ubx[n_states*(N+1)+2: n_states*(N+1) + n_controls*N: n_controls] = v_max     

#Equality constraints g by Multiple Shooting
#All prediction long + Inequality constraints for obstackle.
lbg = ca.DM.zeros(((n_states+n_obstacles)*(N+1),1)) 
ubg = ca.DM.zeros(((n_states+n_obstacles)*(N+1),1))      


#MPC side
lbg[0: n_states*(N+1): 1] = 0
ubg[0: n_states*(N+1): 1] = 0  

#Inequality constraints due to Obstacles
lbg[n_states*(N+1): ((n_states + n_obstacles)*(N+1)): 1] = -ca.inf
ubg[n_states*(N+1): ((n_states + n_obstacles)*(N+1)): 1] = 0         



args ={
       'lbx': lbx, 
       'ubx': ubx, 
       
       'lbg': lbg, # Equality constraints lower bound
       'ubg': ubg, # Equality constraints upper bound        
       } 
#all of the above is just a problem setup
#--------------------------------------------------------------------------

#-------------------------------------------------------------------------
#The simulation loop starts here

t0 = 0
state_init = ca.DM([x_init, y_init, z_init])                # initial state
state_target = ca.DM([x_target, y_target, z_target])        # target state

#DM is mainly used for storing matrices in CasADi
xx = ca.DM(state_init) #xx will contain the history of state
t = ca.DM(t0)

u_init = ca.DM.zeros((n_controls, N))      # initial control, three control inputs
X0 = ca.repmat(state_init, 1, N+1)         # initialize the state full
X0_1 = ca.reshape(X0, n_states*(N+1), 1)

#Start MPC

mpc_iter = 0 #Counter for the loop 

cat_states = DM2Arr(X0)             #Store predicted states
cat_controls = DM2Arr(u_init[:, 0]) #Store predicted control variables
times = np.array([[0]])

'''
Main Loop
the main simulaton loop... it works as long as the error is greater
than 10^-2 and the number of mpc steps is less than its maximum
value.'''

if __name__ == '__main__':
    main_loop = time()  # return time in sec
    while (ca.norm_2(state_init - state_target) > 0.003) and (mpc_iter < sim_time):
        t1 = time()
        #For initial values, we are putting the inital state and target state to p
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
        # We obtain X values for N horizon.
        t2 = time()
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
        
        #Now, we apply control actions
        t0, state_init, u_init = shift_timestep(T, t0, args["p"][0:3], u, f)
        #state init = Take the next step's (now current) states
        #u_init = New initilaziton of optimization variables.
        #t0 = next time step
        
        t = np.vstack((  
            t,
            t0
        ))
        
        X0 = ca.horzcat(
            X0[:, 1:],
            ca.reshape(X0[:, -1], -1, 1)
        )
        # xx ...
        print(mpc_iter)
        print(t2-t1) #analysis for HW implemention 
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
        
 
    #Animation
    #TrajectoryAnimate(cat_states, t, np.array([x_init, y_init, z_init, x_target, y_target, z_target]), obs_x, obs_y, obs_z, obs_dim)