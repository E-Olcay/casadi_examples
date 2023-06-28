# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 10:47:06 2022

@author: olcay
"""


from time import time
import casadi as ca
import numpy as np
from casadi import sin, cos, tan, atan, pi
from plot import draw_trajectory, draw_control_actions
#---------------------------GENERAL SETTINGS ----------------------------------------------#

#MPC settings
T = 0.2 #samplig time
N = 15  #prediction horizon Nx0.2

#Constant parameters
l_f = 0.5
l_r = 0.5

# state weights matrix (Q_X, Q_Y, Q_V, Q_psi)
Q_x = 1
Q_y = 1 
Q_v = 1
Q_psi = 4


# controls weights matrix
R1 = 1
R2 = 1

# init and target states
x_init = 0
y_init = 0
v_init = 0
psi_init = 0

x_target = 10
y_target = 10
v_target = 0
psi_target = 0 #pi/2

    

#control variables restriction (u1, u2)
u1_max = 2
u1_min = -2
u2_max = pi/4
u2_min = -u2_max

#State constraints
x_lb = -400
y_lb = -400
v_lb = -5
psi_lb = -ca.inf

x_ub = 400
y_ub = 400
v_ub = 5
psi_ub = ca.inf


#Simulation Settings
sim_time = 180



#-------------------------------------------END OF GENERAL SETTINGS ---------------------------
#shift function for simulation
#shift to the next time step. 
def shift_timestep(T, t0, state_init, u, f): 
    f_value = f(state_init, u[:, 0])         #Calculate next states ??how you can add distrubance?
    next_state = ca.DM.full(state_init + (T * f_value))

    t0 = t0 + T
    u_init = ca.horzcat(
        u[:, 1:], #trim the first entry! (because we have allready used it) 
        ca.reshape(u[:, -1], -1, 1) #repeat the last one
    )
    #in the first step, we initalized our control variable with 0,0
    #After calculating the first u
    #we can finally initalize with the last u (args['x0'] = ca.reshape(u0, n_controls*N, 1) )
    return t0, next_state, u_init

def DM2Arr(dm):
    return np.array(dm.full())

#State variables 
x=ca.SX.sym("x") #as symbolic variable
y=ca.SX.sym("y") 
v=ca.SX.sym("v")
psi=ca.SX.sym("psi")

states = ca.vertcat(x,y,v,psi)
n_states = states.numel()

#Control variables
u1 = ca.SX.sym("u1")
u2 = ca.SX.sym("u2")

controls = ca.vertcat(u1,u2)
n_controls = controls.numel()

# = Euler Discretization = (can be replaced by RK4)

# x(k+1) = x(k) + delta_t*v(k)*cos(psi + beta(u2))
# y(k+1) = y(k) + delta_t*v(k)*sin(psi + beta(u2))
# v(k+1) = v(k) + delta_t*u1
# psi(k+1) = psi(k) + delta_t*(v(k)/l_r)*sin(beta(u2))

# beta = atan(tan(u2)*(l_f/(l_r + l_f)))

rhs = ca.vertcat(v*cos(psi + atan(tan(u2)*(l_f/(l_r + l_f)))), #state space - right hand system
       v*sin(psi + atan(tan(u2)*(l_f/(l_r + l_f)))),
       u1,
       (v/l_r)*sin(atan(tan(u2)*(l_f/(l_r + l_f)))))

f = ca.Function("f", [states,controls],[rhs]) #f(states,controls) = rhs

U = ca.SX.sym("U", n_controls, N) #u = n_controls x N matrix
#optimization problem parameters 
P = ca.SX.sym("P",n_states + n_states) #Inıtial state and Reference State
X = ca.SX.sym("X",n_states,N+1)        #4(Initial + Current)States Matrix

# state weights matrix (Q_X, Q_Y, Q_THETA)
Q = ca.diagcat(Q_x, Q_y, Q_v, Q_psi)
# controls weights matrix
R = ca.diagcat(R1, R2)


#Given P (Inıtial Position and target state (never change))
#we will calculate next states with the control varibles and the current state.

#According to our calculation, next (N) states in every +T time

# X[:,0] = P[0:3] #take updated current state from P

g = X[:, 0] - P[:n_states] #initial condition constraints

#All are symbolic,  we will call new prediction in later in the loop

#MPC Contrloller (Running State Controllers)
#-----------------------------------------------------------------------------------------
#(Online Optimization) for every time step
#We want to see how close we are to our target state after prediction
#Objective Function l(x,u)=  (||x_u - x^r||^2) *Q  +  (||u-u^r||^2) *R
#u_r = 0, u = currently used 
#x_r = target, at least same for N iterations (Here, prediction horizon = control horizon)

#in basicoptimization.py we solved optimal control problem with NLP
#Minimized objective functions which is: obj = x**2 - 6 * x + 13 #calculate obj func.
#Objective function is sum of all iterations, so start with 0
#-----------------------------------------------------------------------------------------
#Now similar to Optimal Control NLP solver, but we have objecktive function from MPC
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
    st_next_euler = st + (T * f_value)         
    g = ca.vertcat(g, st_next - st_next_euler) #Equality constraint

        

#recap how to step in OPC
# x = SX.sym("w") #as a symbolic variable
# obj = x**2 - 6 * x + 13 #calculate obj func.
# g = [] #optimization constraints is empty
# p = [] #optimization problem parameters - empty (no parameter)


#X and U global variables, when changing inside the loop (simulation)
#change also here. 

#__________MAIN DIFFERENCE : OBSTACLES IN EQUALITY CONSTRAINTS_________________
#We can add obstacles coinstraints to our equality coinstraints (g)

    
#First part of g, we put all MPC equlity constraints. We can add also inequality constraints.

#in multiple shooting, we will put g for equality constraints. 
#In Multiple Shooting, we have x as a control varaible. Then we can actually
#find the difference between horizons.
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
        'acceptable_obj_change_tol': 1e-6 #acceptable change on objective function
    },
    'print_time': 0
}

solver = ca.nlpsol('solver', 'ipopt',nlp_prob,opts)

#For all prediction long.2xN+1 States, 2XN for  controls
lbx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1)) #zero matrix
ubx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))

#states side
#[start:end:step]
lbx[0: n_states*(N+1): n_states] = x_lb         # x lower bound 
lbx[1: n_states*(N+1): n_states] = y_lb         # y lower bound
lbx[2: n_states*(N+1): n_states] = v_lb         # v lower bound
lbx[3: n_states*(N+1): n_states] = psi_lb       # psi lower bound

ubx[0: n_states*(N+1): n_states] = x_ub         # x upper bound
ubx[1: n_states*(N+1): n_states] = y_ub         # y upper bound
ubx[2: n_states*(N+1): n_states] = v_ub         # v upper bound
ubx[3: n_states*(N+1): n_states] = psi_ub       # psi upper bound

#Control Side
lbx[n_states*(N+1):(n_states*(N+1) + n_controls*N):n_controls] = u1_min #(-0,6)        # u1 lower bound 
lbx[(n_states*(N+1))+1:(n_states*(N+1) + n_controls*N):n_controls] = u2_min #(-0,78)   # u2 lower bound 

ubx[n_states*(N+1):(n_states*(N+1) + n_controls*N):n_controls] = u1_max                # u2 upper bound
ubx[n_states*(N+1)+1: n_states*(N+1) + n_controls*N: n_controls] = u2_max              # u2 upper bound

#Equality constraints g by Multiple Shooting
#All prediction long + Inequality constraints for obstackle.
lbg = ca.DM.zeros(((n_states)*(N+1),1)) 
ubg = ca.DM.zeros(((n_states)*(N+1),1))

#MPC side
lbg[0: n_states*(N+1): 1] = 0
ubg[0: n_states*(N+1): 1] = 0

#Obstacle Side, inequality constraints
#lbg[n_states*(N+1): ((n_states + n_obstacles)*(N+1)): 1] = -ca.inf
#ubg[n_states*(N+1): ((n_states + n_obstacles)*(N+1)): 1] = 0 #this couldnt be postive

#Inequality constraints due to Obstackles

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
state_init = ca.DM([x_init, y_init, v_init, psi_init])            # initial state
state_target = ca.DM([x_target, y_target, v_target, psi_target])  # target state

#DM is mainly used for storing matrices in CasADi
xx = ca.DM(state_init) #xx will contain the history of state
t = ca.DM(t0)


u_init = ca.DM.zeros((n_controls, N))      # initial control (two control inputs)
X0 = ca.repmat(state_init, 1, N+1)         # initial state full
X0_1 = ca.reshape(X0, n_states*(N+1), 1)


#Start MPC

mpc_iter = 0 #Counter for the loop 

cat_states = DM2Arr(X0) #Store predicted states
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
        
        #in this loop, the only change first p (initial,target) then x0
        #Every time step,  the initial, current (state_init) will change
        #state target remains the same.
        #First step we are initalizing optimization variables with zero!
        
        #Now, "sol" have two parameters
        #sol.x -> my minimizer for the object function ( Control Variables)
        #Optimal Control Variables as u
        
        u = ca.reshape(sol['x'][n_states * (N + 1):], n_controls, N)
        X0 = ca.reshape(sol['x'][: n_states * (N+1)], n_states, N+1)
        #Compute Optimal solution trajectory
        # new_prediction_value =new_prediction(u,args["p"])
        #We will recieve X values for N horizon.
       
        cat_states = np.dstack(( #Stack arrays in sequence depth wise (along third axis).
            cat_states,     #old one
            DM2Arr(X0)      #new one 
        ))
        #It saved horizontally
        
        #storage our used control value 2x1
        cat_controls = np.vstack(( #Stack arrays in sequence vertically (row wise).
            cat_controls,  #old one stored here
            DM2Arr(u[:, 0]) #new one
        ))
        #storage all u (predicted and currently), predicted = will used next step
        #It saved vertically
        
        
        #Now next step, we applied control action;
        t0, state_init, u_init = shift_timestep(T, t0, args["p"][0:4], u, f)
        #state init = Take the next step's (now current) states
        #u_init = New initilaziton of the optimization variables.
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
        print(t2-t1) #Check for HW implemention 
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
    
            

    #Plot trajectory
    draw_trajectory(cat_states)
    #Plot control actions
    draw_control_actions(cat_controls,t)
   



    





