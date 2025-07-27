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

gp_outputs = 2 # '2', '4' number of outputs of the GP model (p, theta, v, w)
include_history = False # if True, the GP model will include the history of the last two states and inputs
use_simsolver = True # if True, the simulation will be done with the AcadosSimSolver integrator, otherwise with the manual discrete function

param_tau = 0.03
param_l_nom = 0.5
param_l_real = 0.5

ts_real = 0.01 # [s] real simulation integration step
integration_steps_ratio = 1 # ratio between real integration step and mpc integration step
ts_mpc = integration_steps_ratio * ts_real # [s] mpc integration step
time_horizon_mpc = 2 # [s] mpc time horizon
n_steps_mpc_horizon = int(time_horizon_mpc / ts_mpc) # number of steps in the mpc prediction horizon

simulation_time = 3 # [s] duration of the simulation
sim_steps = int(simulation_time / ts_real) # number of steps in the simulation

Q = np.diagflat([10.0, 10.0, 0.001, 0.001])  # [cart, theta, cart_vel, omega]
R = np.array([[0.01]])                    # [u]
Qe = np.diagflat([10.0, 10.0, 0.001, 0.001])  # terminal cost

#Q = np.diagflat([15.0, 10.0, 0.9, 0.1])  # [cart, theta, cart_vel, omega]
#R = np.array([[0.1]])                    # [u]
#Qe = 10*np.diagflat([10.0, 10.0, 0.1, 0.1])  # terminal cost

setup_dict = {
    "param_tau" : param_tau,
    "param_l_nom" : param_l_nom,
    "param_l_real" : param_l_real,
    "ts_real" : ts_real,
    "integration_steps_ratio" : integration_steps_ratio, 
    "ts_mpc" : ts_mpc,
    "time_horizon_mpc" : time_horizon_mpc,
    "n_steps_mpc_horizon" : n_steps_mpc_horizon,
    "Q" : Q, 
    "R" : R, 
    "Qe" : Qe,    
}

# Definition of Acados OCP Options
#ocp_options = AcadosOcpOptions()
#ocp_options.N_horizon = n_steps_mpc_horizon
#ocp_options.Tsim = ts_mpc
#ocp_options.tf = time_horizon_mpc
#ocp_options.gloalization = '' # 'FIXED_STEP', 'MERIT_BACKTRACKING'
#ocp_options.tol

delete_json_files()

real_model = export_pendulum_ode_model(ts_real) # standard model that is simulated, absence of disturbances
ctrl_model = export_pendulum_ode_model(ts_mpc) # model for the MPC controller, with the same dynamics as the real model (until now)

# Definition of the OCP problem, which will be solved by the mpc controller 
ocp = export_ocp_cartpendulum(N=n_steps_mpc_horizon, T=time_horizon_mpc, model=ctrl_model, integrator_type = 'ERK', Q=Q, R=R, Qe=Qe)
ocp.parameter_values = np.array([param_l_nom]) # nominal value of the pednulum length
#ocp.solver_options = ocp_options
ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')  # acados ocp solver

# AcadosSim object for simulating the real system
# This is used to simulate the system with the optimal control input obtained from the OCP solver
# It uses the real model with the real parameters
sim = AcadosSim()
sim.model = real_model
sim.parameter_values = np.array([param_l_real])  # real value of the pendulum length
sim.solver_options.integrator_type= 'ERK'
sim.solver_options.T = ts_real
sim_solver = AcadosSimSolver(sim, json_file='acados_sim_nominal.json')
sim_solver.model = real_model
sim_solver.set("p", np.array([param_l_real]))

# Set the initial state of the simulation and assign it to the SimSolver object
x0 = np.array([0.0, np.pi, 0.0, 0.0]) #initial state
x_current = x0.copy()
sim_solver.set("x", x0)

# Initializations of the lists to store the simulation results
X_sim = [x0.copy()]
X_gp = [] # list to store the inputs for the GP model in the case we store the history of states and inputs
U_sim = []
next_pred_state = [] # list to store the predicted next state from the OCP solver

print(f"\n\n\nPrediction horizon is {time_horizon_mpc} [s]")
print(f"Number of steps in the horizon is {n_steps_mpc_horizon}")
print(f"Integration step is {ts_mpc} [s]")
print(f"Length of the total simulation is {simulation_time} [s]")
print(f"Number of simulation steps is {sim_steps}")

u0 = np.zeros((1,))

# Definition of the manual discrete integrator if we don't use the AcadosSimSolver
lin_disc_dyn_bar, f_disc_dyn_bar = export_discrete_integrator(export_pendulum_ode_model(ts_real))

input("\n\nPress Enter to start the simulation... ")
print("Starting simuation...")
for i in range(sim_steps):
    print("---------------------------------------------------\n\n\n")
    print(f"Iteration: {i+1}/{sim_steps}, time: {i*ts_real:.3f}s")
    print("x_current before solving OCP: ", x_current)

    #ocp_solver.set(0, "x", x_current)
    ocp_solver.set(0, "lbx", x_current)
    ocp_solver.set(0, "ubx", x_current)

    # Solve the OCP problem
    ocp_solver.solve()

    # Get OCP statistics
    print(
      f"CPT: {ocp_solver.get_stats('time_tot')*1000:.2f}ms |\n "
      f"Shooting (linearization): {ocp_solver.get_stats('time_lin')*1000:.2f}ms |\n "
      f"QP Solve: {ocp_solver.get_stats('time_qp_solver_call')*1000:.2f}ms |\n "
      f"Opt. Crit: {ocp_solver.get_stats('residuals')[0]:.3e} |\n "
      f"SQP Iter: {ocp_solver.get_stats('sqp_iter')}")
    print("x_current after solving OCP: ", x_current)
    
    # Get the predicted state at the next time step
    act_state = ocp_solver.get(0, "x")
    pred_state = ocp_solver.get(1, "x")
    print(f"actual state is {act_state}, while next predicted is {pred_state}")
    # Get the optimal control input and the predicted state
    u0 = ocp_solver.get(0, "u")
    print("Optimal control input is ", u0)

    U_sim.append(u0.copy())
    next_pred_state.append(pred_state.copy())

    if use_simsolver:
        print("USING ACADOSSIMSOLVER FOR SIMULATION")
        # Simulate the system with the optimal control input
        sim_solver.set("x", x_current)
        print("Just set x_current in the simulation solver: ", x_current)
        sim_solver.set("u", u0)
        sim_solver.solve()
        # Get the next state of the simulation
        x_next = sim_solver.get("x")
    elif use_simsolver == False:
        # Discrete manual integrator 
        print("INTEGRATING MANUALLY")
        x_next = f_disc_dyn_bar(x_current, u0, np.array([param_l_real])).full().flatten()
    #Update x_current
    x_current = x_next.copy()

    print("x_next from real simulation: ", x_next)    
    # Store the states and inputs of the simulation
    print("x_current shape is ", x_current.shape)
    X_sim.append(x_current.copy())
print("End of the simulation loop")


print("Length of X_sim is ", len(X_sim))
# Convert lists to numpy arrays
X_sim = np.array(X_sim[:-1])
X_gp = np.array(X_gp[:-1])
U_sim = np.array(U_sim)
next_pred_state = np.array(next_pred_state)   # [p, theta, v, omega and all the past states]
print("X_sim shape is ", X_sim.shape)
print("next_pred_state vector shape is ", next_pred_state.shape)

# GAUSSIAN PROCESSES TARGETS 
# Black Box model: target is the difference between the next and the actual state
Y_gp_p =     X_sim[1:, 0]  - X_sim[:-1, 0] 
Y_gp_theta = X_sim[1:, 1]  - X_sim[:-1, 1] 
Y_gp_v =     X_sim[1:, 2]  - X_sim[:-1, 2] 
Y_gp_w =     X_sim[1:, 3]  - X_sim[:-1, 3] 

# Time vector of the simulation for plotting
time_vec = np.linspace(0, simulation_time, sim_steps)
# Plot the results
plt.figure(figsize=(10, 6))
# First subplot
plt.subplot(3, 1, 1)
plt.plot(time_vec, X_sim[:, 0], label='Cart Position')
plt.plot(time_vec, X_sim[:, 1], label='Pendulum Angle')
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
plt.ylabel('velocity (m/s, rad/s)')
plt.legend()
plt.savefig("nominal_data_collecting.svg")

## Plot the results
#plt.figure(figsize=(10, 6))
## First subplot
#plt.subplot(3, 1, 1)
#plt.plot(time_vec, X_sim[:, 0] - next_pred_state[:, 0], label='Cart Position diff sim-pred')
#
#plt.xlabel('time (s)')
#plt.ylabel('angle (rad) / position (m)')
#plt.legend()
#plt.grid()
## Second subplot
#plt.subplot(3, 1, 2)
#plt.plot(time_vec, X_sim[:, 1] - next_pred_state[:, 1], label='Pendulum Angle diff sim-pred ')
#plt.grid()
#plt.xlabel('time (s)')
#plt.ylabel('force (N)')
#plt.legend()
## Third subplot
#plt.subplot(3, 1, 3)
#plt.plot(time_vec, X_sim[:, 2] - next_pred_state[:, 2], label='cart velocity diff sim-pred')
#plt.plot(time_vec, X_sim[:, 3] - next_pred_state[:, 3], label='pendulum velocity diff sim-pred')
#plt.grid()
#plt.xlabel('time (s)')
#plt.ylabel('velocity (m/s, rad/s)')
#plt.legend()
#plt.savefig("nominal_sim_pred_diff.svg")
plt.show()

visualize_inverted_pendulum(X_sim, U_sim, time_vec, ts_real)

# Make sure all target vectors are reshaped as column vectors
Y_gp_p = Y_gp_p.reshape(-1, 1)
Y_gp_theta = Y_gp_theta.reshape(-1, 1)
Y_gp_v = Y_gp_v.reshape(-1, 1)
Y_gp_w = Y_gp_w.reshape(-1, 1)

if gp_outputs == 2:
    # Combine GP targets
    Y_gp_all = np.hstack([Y_gp_v, Y_gp_w])
    print("GP targets shape is ", Y_gp_all.shape)
elif gp_outputs == 4:
    # Combine GP targets
    Y_gp_all = np.hstack([Y_gp_p, Y_gp_theta, Y_gp_v, Y_gp_w])
    print("GP targets shape is ", Y_gp_all.shape)    

if include_history == False:
    # Save X_sim + targets
    filename_sim_targets = "nominal.csv"
    X_sim_and_targets = np.hstack([X_sim[:-1], U_sim[:-1], Y_gp_all])
    print("GP input shape is ", X_sim[:-1].shape)
    np.savetxt(filename_sim_targets, X_sim_and_targets, delimiter=',', header='x1,x2,x3,x4,u,Y_v,Y_w', comments='')

filename_npz = "nominal_no_actuation.npz"
np.savez(filename_npz, X_sim=X_sim, U_sim=U_sim, setup_dict=setup_dict)

max_abs_scale(filename_sim_targets)

print("Image saved as nominal_data_collecting.svg")
print(f"Data saved in: \n   {filename_sim_targets} and in {filename_npz}")
print("Normalized data saved in nominal_normalized:csv")