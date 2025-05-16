import sys, os, time

sys.path += ["../../external/"]


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

from my_pendulum_model import *
from utils import *

from casadi_gp_callback import GPDiscreteCallback

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

N_horizon = 60  # number of steps in the horizon
Tf = 1  # time horizon [s]
N_sim = 1000   # numer of simulation steps
Ts_st = Tf / N_horizon

# time_ref, p_ref, theta_ref, v_ref, omega_ref = vertical_points_ref(Ts_st, N_sim, N_horizon)
time_ref, p_ref, theta_ref, v_ref, omega_ref = sinusoidal_ref(Ts_st, N_sim, N_horizon)
# plot_references(time_ref, p_ref, theta_ref, v_ref, omega_ref)

# Definition of AcadosOcpOptions 
ocp_opts = AcadosOcpOptions()
ocp_opts.tf = Tf
ocp_opts.N_horizon = N_horizon 
ocp_opts.qp_solver = "FULL_CONDENSING_HPIPM"


nominal_model = export_my_augmented_pendulum_ode_model_with_discrete_rk4(Ts_st, black_box=False) # with black_box false w do not disable the dynamics for v and omega
# print("nominal_model.p is ", nominal_model.p)

real_model = export_my_augmented_pendulum_ode_model_with_discrete_rk4(Ts_st)
# real_model.set("p", 0.5)
# print("real_model.p is ", real_model.p)

ocp = export_ocp_cartpendulum_discrete_memory(N_horizon, Tf, nominal_model)   
ocp.parameter_values = np.array([0.8])
# print("ocp.model.p is ", ocp.model.p)
ocp.solver_options = ocp_opts
ocp.solver_options.nlp_solver_tol_eq = 1e-2
ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")

sim = AcadosSim()
sim.model = real_model
sim.parameter_values = np.array([0.5])
sim.solver_options.T = Tf / N_horizon

sim_solver = AcadosSimSolver(sim, json_file="acados_sim.json")
sim_solver.model = real_model
# Set real model parameter differeent from the nominal model
sim_solver.set("p", 0.5)
# print("sim_solver.model.p is ", sim_solver.model.p)


x0 = np.concatenate((np.array([0.0, np.pi, 0.0, 0.0]), np.zeros((8, ))))  # start slightly off upright
# x0 = np.tile(np.array([0.0, np.pi, 0.0, 0.0]), 3)
x_current = x0.copy()

X_sim = [x0.copy()]
U_sim = []
X_gp = np.array([]) # composed by all the states and the input
Y_gp = np.array([]) # difference between next predicted state and the next real state
next_pred_state = [] # next predicted state

u0_vector = np.sin(750*time_ref) * 5

# Simulate the system
for i in range(N_sim):
    # Computation of the optimal control input with nominal model
    ocp_solver.set(0, "lbx", x_current)
    ocp_solver.set(0, "ubx", x_current)
    # Update cost reference at each stage in the horizon
    for stage in range(ocp_solver.N):
        stage_yref = np.array([p_ref[i+stage], theta_ref[i+stage], v_ref[i+stage], omega_ref[i+stage],0, 0, 0, 0, 0, 0, 0, 0, 0])
        ocp_solver.set(stage, "yref", stage_yref)
    # Optionally also set terminal cost reference (if used)
    ocp_solver.set(ocp_solver.N, "yref", np.array([p_ref[i+ocp_solver.N], theta_ref[i+ocp_solver.N], v_ref[i+ocp_solver.N], omega_ref[i+ocp_solver.N], 0, 0, 0, 0, 0, 0, 0, 0]))
    
    ocp_solver.solve()
    # Solve the OCP 
    if i < 100:# or i > 500:
        u0 = np.array([0.0])
    else:
        u0 = ocp_solver.get(0, "u")
        
    pred_state = ocp_solver.get(0, "x")
    
    # print("pred_state is ", pred_state.shape)
    # print("U_sim is ", U_sim.shape)

    # if i < 50:
    #     u0 = 2
    # elif i < 250:
    #     u0 = 0.5
    # elif i < 500:
    #     u0 = -0.5
    # elif i < 800:   
    #     u0 = 0.5
    # else:
    #     u0 = -15
    # u0 = u0_vector[i]

    U_sim.append(u0)
    next_pred_state.append(pred_state)

    # Simulation of the real model
    sim_solver.set("x", x_current)
    sim_solver.set("u", u0)
    sim_solver.solve()

    # Get the next state from the simulation
    x_next = sim_solver.get("x")

    # Update current state
    X_sim.append(x_next.copy())
    x_current = x_next.copy()

# Convert lists to numpy arrays
X_sim = np.array(X_sim)
U_sim = np.array(U_sim)
next_pred_state = np.array(next_pred_state)   # [p, theta, v, omega and all the past states]

# Time vector
time_vec = np.linspace(0, Tf / N_horizon * N_sim, N_sim + 1)

# Test difference between gradient of position vector and the real velocity vector
# plt.figure(figsize=(10, 6))
# plt.plot(time_vec[:], np.gradient(X_sim[:, 1]), label='gradient of position vector')
# plt.plot(time_vec[:], X_sim[:, 3], label='real velocity vector')
# plt.legend()
# plt.grid()
# plt.show()

# Plot the results
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time_vec, X_sim[:, 0], label='Cart Position')
plt.plot(time_vec, X_sim[:, 1], label='Pendulum Angle')
plt.xlabel('time (s)')
plt.ylabel('angle (rad) / position (m)')
plt.legend()
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(time_vec[:-1], U_sim, label='Control Input (Force)')
plt.grid()
plt.xlabel('time (s)')
plt.ylabel('force (N)')
plt.legend()
plt.show()
np.savez("rollout_data.npz", X_sim=X_sim, U_sim=U_sim)

# Plot the difference between the predicted state and the real state
# plt.figure(figsize=(10, 6))
# plt.subplot(2, 1, 1)
# plt.plot(time_vec[:-1], next_pred_state[:,0 ]-X_sim[1:, 0], label='Predicted p - Real p')
# plt.plot(time_vec[:-1], next_pred_state[:,2]-X_sim[1:, 2], label='Predicted v - Real v')
# plt.xlabel('time (s)')
# plt.ylabel('speed (m/s) / position (m)')
# plt.legend()
# plt.grid()
# plt.subplot(2, 1, 2)
# plt.plot(time_vec[:-1], next_pred_state[:,1]-X_sim[1:, 1], label='Predicted theta - Real theta')
# plt.plot(time_vec[:-1], next_pred_state[:,3]-X_sim[1:, 3], label='Predicted omega - Real omega')
# plt.xlabel('time (s)')
# plt.ylabel('angle (rad) / angular speed (rad/s)')
# plt.legend()
# plt.grid()
# plt.show()