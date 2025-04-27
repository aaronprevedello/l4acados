# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: l4acados_dev
#     language: python
#     name: python3
# ---

# + endofcell="--"
# # + metadata={}
import sys, os

sys.path += ["../../external/"]

# # + metadata={}
# %load_ext autoreload
# %autoreload 1
# %aimport l4acados

# # + metadata={}
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

from pendulum_model import (
    export_pendulum_ode_model_with_discrete_rk4,
    export_pendulum_ode_real_model_with_discrete_rk4,
    export_ocp_cartpendulum_discrete,
)
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
    generate_grid_points,
)
from gpytorch import likelihoods
from l4acados.models.pytorch_models.gpytorch_models.gpytorch_gp import (
    BatchIndependentMultitaskGPModel,
)

from l4acados.models.pytorch_models.gpytorch_models.gpytorch_residual_model import (
    GPyTorchResidualModel,
)

N_horizon = 80  # number of steps in the horizon
Tf = 2  # time horizon [s]
N_sim = 350   # numer of simulation steps

# Definition of AcadosOcpOptions 
ocp_opts = AcadosOcpOptions()
ocp_opts.tf = Tf
ocp_opts.N_horizon = N_horizon 
ocp_opts.qp_solver = "FULL_CONDENSING_HPIPM"


nominal_model = export_pendulum_ode_model_with_discrete_rk4(0.025)

real_model = export_pendulum_ode_real_model_with_discrete_rk4(0.025)

ocp = export_ocp_cartpendulum_discrete(N_horizon, Tf, nominal_model)   
ocp.model = nominal_model
ocp.solver_options = ocp_opts
ocp.solver_options.nlp_solver_tol_eq = 1e-2
ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")

sim = AcadosSim()
sim.model = real_model
sim.solver_options.T = Tf / N_horizon

sim_solver = AcadosSimSolver(sim, json_file="acados_sim.json")
sim_solver.model = real_model

x0 = np.array([0.0, 0.0, np.pi, 0.0])  # start slightly off upright
x_current = x0.copy()

X_sim = [x0.copy()]
U_sim = []
X_gp = np.array([]) # composed by all the states and the input
Y_gp = np.array([]) # difference between next predicted state and the next real state
next_pred_state = [] # next predicted state

# Simulate the system
for i in range(N_sim):
    # Computation of the optimal control input with nominal model
    ocp_solver.set(0, "lbx", x_current)
    ocp_solver.set(0, "ubx", x_current)
    ocp_solver.solve()

    u0 = ocp_solver.get(0, "u")
    pred_state = ocp_solver.get(0, "x")
    # print("pred_state is ", pred_state.shape)
    # print("U_sim is ", U_sim.shape)
    
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
next_pred_state = np.array(next_pred_state)

# GENERATE THE GP DATA
# Input to GP
U_sim_padded = np.vstack([U_sim, np.zeros((1, 1))])
X_gp = np.hstack([X_sim, U_sim_padded])

# print("X_gp shape is ", X_gp.shape)
# print("next_pred_state shape is ", next_pred_state.shape)

# Targets 
# Grey Box model: target is the difference between the next predicted state and the next real state   
Y_gp_v = X_gp[1:, 1] - next_pred_state[:, 1]
Y_gp_w = X_gp[1:, 3] - next_pred_state[:, 3]
# Black Box model: target is the difference between the next and the actual state
# Y_gp_v = X_gp[1:, 1] - X_gp[:-1, 1]
# Y_gp_w = X_gp[1:, 3] - X_gp[:-1, 3]

# Time vector
time = np.linspace(0, Tf / N_horizon * N_sim, N_sim + 1)

# Plot the results
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time, X_sim[:, 0], label='Cart Position')
plt.plot(time, X_sim[:, 2], label='Pendulum Angle')
plt.xlabel('time (s)')
plt.ylabel('angle (rad) / position (m)')
plt.legend()
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(time[:-1], U_sim, label='Control Input (Force)')
plt.grid()
plt.xlabel('time (s)')
plt.ylabel('force (N)')
plt.legend()
plt.show()
np.savez("rollout_data.npz", X_sim=X_sim, U_sim=U_sim)



# Define the GP model
train_inputs = torch.tensor(X_gp[:-1, :], dtype=torch.float32)  
train_outputs = torch.tensor(np.hstack([Y_gp_v.reshape(-1, 1), Y_gp_w.reshape(-1, 1)]), dtype=torch.float32)
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(2)
gp_model = BatchIndependentMultitaskGPModel(
    train_x = train_inputs,
    train_y = train_outputs,
    input_dimension=5,
    residual_dimension=2,
    likelihood=likelihood,
)

# Train the GP model on the data   
gp_model, likelihood = train_gp_model(
    gp_model, training_iterations=300)


save_path = "gp_model.pth"
# Save state dicts and training data (optional if not used later)
torch.save({
    'model_state_dict': gp_model.state_dict(),
    'likelihood_state_dict': likelihood.state_dict(),
    'train_x': gp_model.train_inputs[0],  # if needed
    'train_y': gp_model.train_targets,    # if needed
}, save_path)

# Load state dicts
checkpoint = torch.load('gp_model.pth')
gp_model.load_state_dict(checkpoint['model_state_dict'])
likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
gp_model.eval()
likelihood.eval()

# Define residual model
res_model = GPyTorchResidualModel(gp_model)

# Define mapping from gp outputs (dim=2) to model states (dim=4)
Ts_st = Tf / N_horizon
B = np.array([
    [Ts_st / 2, 0],
    [0, Ts_st / 2],
    [1.0, 0],
    [0, 1.0],
])
# Define l4acados solver
l4acados_solver = ResidualLearningMPC(
    ocp=ocp,
    residual_model=res_model,
    B = B
)

# New simulation using the residual model
nominal_model = export_pendulum_ode_model_with_discrete_rk4(0.025)

real_model = export_pendulum_ode_real_model_with_discrete_rk4(0.025)

ocp = export_ocp_cartpendulum_discrete(N_horizon, Tf, nominal_model)   
ocp.model = nominal_model
ocp.solver_options.nlp_solver_tol_eq = 1e-2
ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")

sim = AcadosSim()
sim.model = real_model
sim.solver_options.T = Tf / N_horizon

sim_solver = AcadosSimSolver(sim, json_file="acados_sim.json")
sim_solver.model = real_model

x0 = np.array([0.0, 0.0, np.pi, 0.0])  # start slightly off upright
x_current = x0.copy()

X_sim_res = [x0.copy()]
U_sim_res = []
X_gp = np.array([]) # composed by all the states and the input
Y_gp = np.array([]) # difference between next predicted state and the next real state


# Simulate the system with the residual model
for i in range(N_sim):
    # Computation of the optimal control input with nominal model + residual model
    print("Second simulation iteration ", i)
    l4acados_solver.set(0, "lbx", x_current)
    l4acados_solver.set(0, "ubx", x_current)
    l4acados_solver.solve()

    u0 = l4acados_solver.get(0, "u")
    # print("u0 is ", u0.shape)
    # print("U_sim is ", U_sim.shape)
    
    U_sim_res.append(u0)

    # Simulation of the real model
    sim_solver.set("x", x_current)
    sim_solver.set("u", u0)
    sim_solver.solve()

    # Get the next state from the simulation
    x_next = sim_solver.get("x")

    # Update current state
    X_sim_res.append(x_next.copy())
    x_current = x_next.copy()
print("End of the simulation")


# Convert lists to numpy arrays
X_sim_res = np.array(X_sim_res)
U_sim_res = np.array(U_sim_res)


# Generate the GP data
# Input to GP
U_sim_padded = np.vstack([U_sim_res, np.zeros((1, 1))])
X_gp = np.hstack([X_sim, U_sim_padded])
# Targets    
Y_gp_v = X_gp[1:, 1] - X_gp[:-1, 1]
Y_gp_w = X_gp[1:, 3] - X_gp[:-1, 3]

# Time vector
time = np.linspace(0, Tf / N_horizon * N_sim, N_sim + 1)

# Plot the results
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(time, X_sim_res[:, 0], label='Res Cart Position')
plt.plot(time, X_sim[:, 0], label='Cart Position')
plt.xlabel('time (s)')
plt.ylabel('position (m)')
plt.legend()
plt.grid()
plt.subplot(3, 1, 2)
plt.plot(time[:-1], U_sim_res, label='Res Control Input (Force)')
plt.plot(time[:-1], U_sim, label='Control Input (Force)')
plt.grid()
plt.xlabel('time (s)')
plt.ylabel('force (N)')
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(time, X_sim_res[:, 2], label='Res Pendulum Angle')
plt.plot(time, X_sim[:, 2], label='Pendulum Angle')
plt.ylabel('angle (rad)')
plt.xlabel('time (s)')
plt.legend()
plt.grid()
plt.show()

np.savez("rollout_data_res_ctrl.npz", X_sim_res=X_sim_res, U_sim_res=U_sim_res)
