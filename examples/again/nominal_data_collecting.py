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

ref_type = 'swing_up' # 'swing_up', 'vertical', 'sin, 
param_tau = 0.05
param_l_nom = 0.5
param_l_real = 0.5

ts_real = 0.025 # [s] real simulation integration step
integration_steps_ratio = 1 # ratio between real integration step and mpc integration step
ts_mpc = integration_steps_ratio * ts_real # [s] mpc integration step
time_horizon_mpc = 1.5 # [s] mpc time horizon
n_steps_mpc_horizon = int(time_horizon_mpc / ts_mpc) # number of steps in the mpc prediction horizon

simulation_time = 5 # [s] duration of the simulation
sim_steps = int(simulation_time / ts_real) # number of steps in the simulation

# Definition of Acados OCP Options
#ocp_options = AcadosOcpOptions()
#ocp_options.N_horizon = n_steps_mpc_horizon
#ocp_options.Tsim = ts_mpc
#ocp_options.tf = time_horizon_mpc
#ocp_options.gloalization = '' # 'FIXED_STEP', 'MERIT_BACKTRACKING'
#ocp_options.tol

real_model = export_pendulum_ode_model() # standard model
ctrl_model = export_pendulum_ode_model()

ocp = export_ocp_cartpendulum(N=n_steps_mpc_horizon, T=time_horizon_mpc, model=ctrl_model, integrator_type = 'ERK')
ocp.parameter_values = np.array([param_l_nom])
#ocp.solver_options = ocp_options
ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')

sim = AcadosSim()
sim.model = real_model
sim.parameter_values = np.array([param_l_real])
sim.solver_options.integrator_type= 'ERK'
sim.solver_options.T = ts_real

sim_solver = AcadosSimSolver(sim, json_file='acados_sim_nominal.json')
sim_solver.model = real_model
sim_solver.set("p", np.array([param_l_real]))

x0 = np.array([0.0, np.pi, 0.0, 0.0]) #initial state
x_current = x0.copy()
sim_solver.set("x", x0)

X_sim = [x0.copy()]
U_sim = []
next_pred_state = []
time_inst = [0.0]

print(f"Prediction horizon is {time_horizon_mpc} [s]")
print(f"Number of steps in the horizon is {n_steps_mpc_horizon}")
print(f"Integration step is {ts_mpc} [s]")
print(f"Length of the total simulation is {simulation_time} [s]")
print(f"Number of simulation steps is {sim_steps}")

u0 = np.zeros((1,))

print("Starting simuation...")
for i in range(sim_steps):
    print("---------------------------------------------------\n\n\n")
    print("Iteration: ", i)
    print("x_current before solving OCP: ", x_current)

    #ocp_solver.set(0, "x", x_current)
    ocp_solver.set(0, "lbx", x_current)
    ocp_solver.set(0, "ubx", x_current)

    ocp_solver.solve()

    # Get OCP statistics
    print(
      f"CPT: {ocp_solver.get_stats('time_tot')*1000:.2f}ms |\n "
      f"Shooting (linearization): {ocp_solver.get_stats('time_lin')*1000:.2f}ms |\n "
      f"QP Solve: {ocp_solver.get_stats('time_qp_solver_call')*1000:.2f}ms |\n "
      f"Opt. Crit: {ocp_solver.get_stats('residuals')[0]:.3e} |\n "
      f"SQP Iter: {ocp_solver.get_stats('sqp_iter')}")
    print("x_current after solving OCP: ", x_current)
    
    act_state = ocp_solver.get(0, "x")
    pred_state = ocp_solver.get(1, "x")
    print(f"actual state is {act_state}, while next predicted is {pred_state}")
    # Get the optimal control input and the predicted state
    u0 = ocp_solver.get(0, "u")

    U_sim.append(u0.copy())
    next_pred_state.append(pred_state.copy())

    # Simulate the ystem with the optimal control input
    sim_solver.set("x", x_current)
    print("Just set x_current in the simulation solver: ", x_current)
    sim_solver.set("u", u0)
    sim_solver.solve()

    # Get tje next state of the simulation
    x_next = sim_solver.get("x")
    print("x_next from real simulation: ", x_next)
    #Update x_current
    x_current = x_next.copy()

    # Store the states and inputs of the simulation
    print("x_current shape is ", x_current.shape)
    X_sim.append(x_current.copy())

print("Length of X_sim is ", len(X_sim))
# Convert lists to numpy arrays
X_sim = np.array(X_sim[:-1])
U_sim = np.array(U_sim)
next_pred_state = np.array(next_pred_state)   # [p, theta, v, omega and all the past states]
print("X_sim shape is ", X_sim.shape)
# TARGETS 
# Black Box model: target is the difference between the next and the actual state
Y_gp_p = X_sim[1:, 0] - X_sim[:-1, 0]
Y_gp_theta = X_sim[1:, 1] - X_sim[:-1, 1]
Y_gp_v = X_sim[1:, 2] - X_sim[:-1, 2]
Y_gp_w = X_sim[1:, 3] - X_sim[:-1, 3]

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
plt.show()

visualize_inverted_pendulum(X_sim, U_sim, time_vec)

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

print(f"Data saved in: \n{filename_sim_targets}")