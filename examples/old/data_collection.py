import sys, os, time

sys.path += ["../../external/"]

from collections import deque
import numpy as np
from scipy.stats import norm
import casadi as cas
from acados_template import (
    AcadosOcp,
    AcadosSim,
    AcadosSimSolver,
    AcadosOcpSolver,
    AcadosOcpOptions,
    ZoroDescription,
)
import matplotlib.pyplot as plt
import torch
import gpytorch
import copy

# zoRO imports
import l4acados
from l4acados.controllers import (
    ResidualLearningMPC,
)

from pendulum_model import *
from utils import *

# gpytorch_utils
from gpytorch_utils.gp_hyperparam_training import (
    generate_train_inputs_acados,
    generate_train_outputs_at_inputs,
    train_gp_model,
)
from gpytorch_utils.gp_utils import (
    gp_data_from_model_and_path,
    gp_derivative_data_from_model_and_path,
    plot_gp_data,
    generate_grid_points,)


from gpytorch import likelihoods
from l4acados.models.pytorch_models.gpytorch_models.gpytorch_gp import (
    BatchIndependentMultitaskGPModel,
)

from l4acados.models.pytorch_models.gpytorch_models.gpytorch_residual_model import (
    GPyTorchResidualModel,
)

train_flag = True
ref_type = 'exploratory_ref' # 'swing_up', 'vertical', 'sin', 'mix', 'rich_mix'
actuation_dynamics = 'no' # 'yes', 'no'
param_tau = 0.05
param_l_nom = 0.5
param_l_real = 0.5

N_horizon = 50  # number of steps in the prediction horizon
Tf = 1.5  # time horizon of the mpc prediction [s]
N_ocp = 3000   # numer of simulation steps
Ts_st = Tf / N_horizon  # integrator time step
integration_steps_ratio = 10 # ratio between Ts_st and T_real
Ts_real = Ts_st/integration_steps_ratio # real model integration step
steps_per_mpc = int(Ts_st / Ts_real)
N_real = N_ocp*integration_steps_ratio # number of real steps of the simulation

if ref_type == 'vertical':
    time_ref, p_ref, theta_ref, v_ref, omega_ref = vertical_points_ref(Ts_real, N_real, N_horizon*integration_steps_ratio)
elif ref_type == 'sin':    
    time_ref, p_ref, theta_ref, v_ref, omega_ref = sinusoidal_ref(Ts_real, N_real, N_horizon*integration_steps_ratio)
elif ref_type == 'mix':
    time_ref, p_ref, theta_ref, v_ref, omega_ref = mix_ref(Ts_real, N_real, N_horizon*integration_steps_ratio)
elif ref_type == 'rich_mix':
    time_ref, p_ref, theta_ref, v_ref, omega_ref = rich_mix_ref(Ts_real, N_real, N_horizon*integration_steps_ratio)
elif ref_type == 'exploratory_ref':
    time_ref, u_ref = exploratory_ref(Ts_real, N_real, N_horizon*integration_steps_ratio)
#print("time ref length ", len(time_ref))
#print("p_ref length is ", len(p_ref))
#plot_references(time_ref, p_ref, theta_ref, v_ref, omega_ref)
plt.figure()
plt.plot(time_ref, u_ref)
plt.show()

# Definition of AcadosOcpOptions 
ocp_opts = AcadosOcpOptions()
ocp_opts.tf = Tf
ocp_opts.N_horizon = N_horizon 
ocp_opts.qp_solver = "FULL_CONDENSING_HPIPM"

real_model = export_pendulum_ode_model() # standard model

ocp = export_ocp_cartpendulum_discrete(N_horizon, Tf, real_model, integrator_type="ERK")   
ocp.parameter_values = np.array([param_l_nom])
ocp.solver_options = ocp_opts
ocp.solver_options.nlp_solver_tol_eq = 1e-2
ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")

sim = AcadosSim()
sim.model = real_model
sim.parameter_values = np.array([param_l_real])
sim.solver_options.T = Ts_real

sim_solver = AcadosSimSolver(sim, json_file="acados_sim.json")
sim_solver.model = real_model
# Set real model parameter different from the nominal model
sim_solver.set("p", np.array([param_l_real]))

x0_real = np.array([0.0, np.pi, 0.0, 0.0])
x_current = x0_real.copy()
sim_solver.set("x", x0_real)

X_sim = [x0_real.copy()]
U_sim = []
X_gp = []
Y_gp = np.array([]) # difference between next predicted state and the next real state
next_pred_state = [] # next predicted state

print(f"Prediction horizon is {Tf} [s]")
print(f"Number of steps in the horizon is {N_horizon}")
print(f"Integration step is {Ts_st} [s]")
print(f"Length of the total simulation is {N_real*Ts_real} [s]")
# Buffer to store past two states for p and theta, and past two inputs
# History buffers
state_history = deque([x0_real] * 3, maxlen=3)
input_history = deque([np.zeros(1)] * 3, maxlen=3)
x_next = x0_real.copy() # for the first iteration in order to set u_act as 0
u0 = np.zeros((1,)) # initialize the variable for the optimal control input

# Simulate the system
print("Starting simulation")
for i in range(N_real):

    if i % steps_per_mpc == 0:
        gp_act_input = np.concatenate([state_history[-1], state_history[-2][0:2], input_history[-1], state_history[-3][0:2], input_history[-2]])
    
        X_sim.append(x_current.copy())
        X_gp.append(gp_act_input.copy())
        u0 = np.array([u_ref[i]])
        U_sim.append(u0)
        
    #print("Actual control input is ", u0)
    
    # Simulation of the real model
    sim_solver.set("x", x_current)
    sim_solver.set("u", u0)
    sim_solver.solve()

    # Get the next state from the simulation
    x_next = sim_solver.get("x")       
    
    x_current = x_next.copy()
    
    # Update buffers (exclude u_act for controller's history)
    state_history.append(x_next[0:4].copy())
    input_history.append(u0.copy())

print("Length of X_sim is ", len(X_sim))
# Convert lists to numpy arrays
X_sim = np.array(X_sim[:-1])
U_sim = np.array(U_sim)
next_pred_state = np.array(next_pred_state)   # [p, theta, v, omega and all the past states]

# GENERATE THE GP DATA
# Input to GP
X_gp = np.array(X_gp)

# print("X_gp shape is ", X_gp.shape)
# print("X_sim shape is ", X_sim.shape)
# print("U_sim shape is ", U_sim.shape)

# TARGETS 
# Black Box model: target is the difference between the next and the actual state
Y_gp_p = X_sim[1:, 0] - X_sim[:-1, 0]
Y_gp_theta = X_sim[1:, 1] - X_sim[:-1, 1]
Y_gp_v = X_sim[1:, 2] - X_sim[:-1, 2]
Y_gp_w = X_sim[1:, 3] - X_sim[:-1, 3]

# Time vector
time_vec = np.linspace(0, Tf / N_horizon * N_ocp, N_ocp)  #start,stop,number of points
# Plot the results
plt.figure(figsize=(10, 6))
# First subplot
plt.subplot(2, 1, 1)
plt.plot(time_vec, X_sim[:, 0], label='Cart Position')
plt.plot(time_vec, X_sim[:, 1], label='Pendulum Angle')
#if ref_type != 'swing_up':
#   plt.plot(time_ref[:N_real:10], p_ref[:N_real:10], label='Cart Ref')
#   plt.plot(time_ref[:N_real:10], theta_ref[:N_real:10], label='Pendulum Ref')
plt.xlabel('time (s)')
plt.ylabel('angle (rad) / position (m)')
plt.legend()
plt.grid()
# Second subplot
plt.subplot(2, 1, 2)
plt.plot(time_vec, U_sim, label='Control Input (Force)')
plt.grid()
plt.xlabel('time (s)')
plt.ylabel('force (N)')
plt.legend()
plt.show()

# Make sure all target vectors are reshaped as column vectors
Y_gp_p = Y_gp_p.reshape(-1, 1)
Y_gp_theta = Y_gp_theta.reshape(-1, 1)
Y_gp_v = Y_gp_v.reshape(-1, 1)
Y_gp_w = Y_gp_w.reshape(-1, 1)

#print("X_gp shape is ", X_gp.shape)
#print("X_sim shape is ", X_sim.shape)
#print("Y_gp_p shape is ", Y_gp_p.shape)
#print("Y_gp_theta shape is ", Y_gp_theta.shape)
#print("Y_gp_v shape is ", Y_gp_v.shape)
#print("Y_gp_w shape is ", Y_gp_w.shape)
#print("U_sim shape is ", U_sim.shape)
# Combine GP targets
Y_gp_all = np.hstack([Y_gp_p, Y_gp_theta, Y_gp_v, Y_gp_w])

# Save X_sim + targets
filename_sim_targets = "X_sim_with_gp_targets.csv"
X_sim_and_targets = np.hstack([X_sim[:-1], U_sim[:-1], Y_gp_all])
np.savetxt(filename_sim_targets, X_sim_and_targets, delimiter=',', header='x1,x2,x3,x4,u,Y_p,Y_theta,Y_v,Y_w', comments='')

# Save X_gp + targets
filename_past_states_targets = "input_data/swing_up_past.csv"
X_gp_and_targets = np.hstack([X_gp[:-1], U_sim[:-1], Y_gp_all])
np.savetxt(filename_past_states_targets, X_gp_and_targets, delimiter=',', header='x1,x2,x3,x4,x1_p1,x2_p1,u_p1,x1_p2,x2_p2,u_p2,u,Y_p,Y_theta,Y_v,Y_w', comments='')  # Adjust header manually

print(f"Data saved in: \n{filename_sim_targets}\n{filename_past_states_targets}")