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
ref_type = 'swing_up' # 'swing_up', 'vertical', 'sin', 'mix', 'rich_mix'
actuation_dynamics = 'no' # 'yes', 'no'
param_tau = 0.05
param_l_nom = 0.5
param_l_real = 0.5

Ts_real = 25e-3 # [s] real model integration step
integration_steps_ratio = 1 # ratio between Ts_st and T_real
Ts_st = integration_steps_ratio * Ts_real # integrator time step
Tf = 1.5  # time horizon of the mpc prediction [s]

N_horizon = int(Tf / Ts_st) # number of steps in the horizon

T_sim = 5 # [s]
sim_steps = int(T_sim/Ts_real)  # number of simulation steps

if ref_type == 'vertical':
    time_ref, p_ref, theta_ref, v_ref, omega_ref = vertical_points_ref(Ts_real, sim_steps, N_horizon*integration_steps_ratio)
elif ref_type == 'sin':    
    time_ref, p_ref, theta_ref, v_ref, omega_ref = sinusoidal_ref(Ts_real, sim_steps, N_horizon*integration_steps_ratio)
elif ref_type == 'mix':
    time_ref, p_ref, theta_ref, v_ref, omega_ref = mix_ref(Ts_real, sim_steps, N_horizon*integration_steps_ratio)
elif ref_type == 'rich_mix':
    time_ref, p_ref, theta_ref, v_ref, omega_ref = rich_mix_ref(Ts_real, sim_steps, N_horizon*integration_steps_ratio)
elif ref_type == 'exploratory_ref':
    time_ref, p_ref, theta_ref, v_ref, omega_ref = exploratory_ref(Ts_real, sim_steps, N_horizon*integration_steps_ratio)
#print("time ref length ", len(time_ref))
#print("p_ref length is ", len(p_ref))
#plot_references(time_ref, p_ref, theta_ref, v_ref, omega_ref)

# Definition of AcadosOcpOptions 
ocp_opts = AcadosOcpOptions()
ocp_opts.tf = Tf
ocp_opts.N_horizon = N_horizon 
ocp_opts.Tsim = Ts_st
ocp_opts.qp_solver = "FULL_CONDENSING_HPIPM"

real_model = export_pendulum_ode_model() # standard model

ocp = export_ocp_cartpendulum_discrete(N_horizon, Tf, real_model, integrator_type="ERK")   
ocp.parameter_values = np.array([param_l_nom])
ocp.solver_options = ocp_opts
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
print(f"Length of the total simulation is {T_sim} [s]")
print(f"Number of simulation steps is {sim_steps}")
# Buffer to store past two states for p and theta, and past two inputs
# History buffers
state_history = deque([x0_real] * 3, maxlen=3)
input_history = deque([np.zeros(1)] * 3, maxlen=3)
x_next = x0_real.copy() # for the first iteration in order to set u_act as 0
u0 = np.zeros((1,)) # initialize the variable for the optimal control input

# Simulate the system
print("Starting simulation")
for i in range(sim_steps):
    print("---------------------------------------------------\n\n\n")
    print("Iteration: ", i)
    print("x_current before solving OCP: ", x_current)
    #if i % steps_per_mpc == 0:
    
    #print(f"Computing optimal control input at time step {i*Ts_real}")
    # Computation of the optimal control input with nominal model
    ocp_solver.set(0, "lbx", x_current)
    ocp_solver.set(0, "ubx", x_current)
    # Update cost reference at each stage in the horizon
    if ref_type != 'swing_up':
        for stage in range(ocp_solver.N):
            stage_yref = np.array([p_ref[i+stage], theta_ref[i+stage], v_ref[i+stage], omega_ref[i+stage],0])
            ocp_solver.set(stage, "yref", stage_yref)
        # Optionally also set terminal cost reference (if used)
        ocp_solver.set(ocp_solver.N, "yref", np.array([p_ref[i+ocp_solver.N], theta_ref[i+ocp_solver.N], v_ref[i+ocp_solver.N], omega_ref[i+ocp_solver.N]]))
    # Solve the OCP 
    ocp_solver.solve()
    
    
    print(
      f"CPT: {ocp_solver.get_stats('time_tot')*1000:.2f}ms |\n "
      f"Shooting (linearization): {ocp_solver.get_stats('time_lin')*1000:.2f}ms |\n "
      f"QP Solve: {ocp_solver.get_stats('time_qp_solver_call')*1000:.2f}ms |\n "
      f"Opt. Crit: {ocp_solver.get_stats('residuals')[0]:.3e} |\n "
      f"Statistics: {ocp_solver.get_stats('statistics')} | \n"
      f"SQP Iter: {ocp_solver.get_stats('sqp_iter')}")
    print("x_current after solving OCP: ", x_current)
    act_state = ocp_solver.get(0, "x")
    pred_state = ocp_solver.get(1, "x")
    print(f"actual state is {act_state}, while next predicted is {pred_state}")
    # Get the optimal control input and the predicted state
    u0 = ocp_solver.get(0, "u")
    
    #U_sim.append(u0)
    U_sim.append(u0)
    next_pred_state.append(pred_state)        
    #print("Actual control input is ", u0)
    
    # Simulation of the real model
    sim_solver.set("x", x_current)
    sim_solver.set("u", u0)
    sim_solver.solve()

    # Get the next state from the simulation
    x_next = sim_solver.get("x")       
    print("Next state from the simulation on real model is ", x_next)
    x_current = x_next.copy()
    # Update buffers (exclude u_act for controller's history)
    state_history.append(x_next[0:4].copy())
    input_history.append(u0.copy())

    gp_act_input = np.concatenate([state_history[-1], state_history[-2][0:2], input_history[-1], state_history[-3][0:2], input_history[-2]])
    print("gp_act_input is ", gp_act_input)
    print("state_history is ", state_history)
    print("input_history is ", input_history)
    X_sim.append(x_current.copy())
    X_gp.append(gp_act_input.copy())
    
    

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
time_vec = np.linspace(0, T_sim, sim_steps)  #start,stop,number of points
# Plot the results
plt.figure(figsize=(10, 6))
# First subplot
plt.subplot(3, 1, 1)
plt.plot(time_vec, X_sim[:, 0], label='Cart Position')
plt.plot(time_vec, X_sim[:, 1], label='Pendulum Angle')
if ref_type != 'swing_up':
    plt.plot(time_ref[:sim_steps:10], p_ref[:sim_steps:10], label='Cart Ref')
    #plt.plot(time_ref[:sim_steps:10], theta_ref[:sim_steps:10], label='Pendulum Ref')
plt.xlabel('time (s)')
plt.ylabel('angle (rad) / position (m)')
plt.legend()
plt.grid()
# Second subplot
plt.subplot(3, 1, 2)
plt.plot(time_vec, U_sim, label='Control Input (Force)')
plt.grid()
plt.xlabel('time (s)')
plt.ylabel('force (N)')
plt.legend()
# Third subplot
plt.subplot(3, 1, 3)
plt.plot(time_vec, X_sim[:, 2], label='cart velocity')
plt.plot(time_vec, X_sim[:, 3], label='pendulum velocity')
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
Y_gp_all = np.hstack([Y_gp_v, Y_gp_w])
print("GP targets shape is ", Y_gp_all.shape)

# Save X_sim + targets
filename_sim_targets = "X_sim_with_gp_targets.csv"
X_sim_and_targets = np.hstack([X_sim[:-1], U_sim[:-1], Y_gp_all])
print("GP input shape is ", X_sim[:-1].shape)
np.savetxt(filename_sim_targets, X_sim_and_targets, delimiter=',', header='x1,x2,x3,x4,u,Y_v,Y_w', comments='')

# Save X_gp + targets
filename_past_states_targets = "input_data/swing_up_past.csv"
X_gp_and_targets = np.hstack([X_gp[:-1], U_sim[:-1], Y_gp_all])
#np.savetxt(filename_past_states_targets, X_gp_and_targets, delimiter=',', header='x1,x2,x3,x4,x1_p1,x2_p1,u_p1,x1_p2,x2_p2,u_p2,u,Y_p,Y_theta,Y_v,Y_w', comments='')  # Adjust header manually

print(f"Data saved in: \n{filename_sim_targets}")