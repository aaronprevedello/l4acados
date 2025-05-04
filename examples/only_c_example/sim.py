# NOTE: This version strips out all acados_template imports and uses ctypes to call
# a compiled GP model as the dynamics function inside acados.

import os, sys
import ctypes

sys.path += ["../../external/"]
import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
# from acados_c import *  # low-level acados API

from utils import *
from gpytorch_utils.gp_hyperparam_training import train_gp_model
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from l4acados.models.pytorch_models.gpytorch_models.gpytorch_gp import (
    BatchIndependentMultitaskGPModel,
)

# === Simulation Phase (use real system simulation) ===
from gpytorch import likelihoods
from pendulum_model import (
    export_pendulum_ode_model_with_discrete_rk4,
    export_pendulum_ode_real_model_with_discrete_rk4,
    export_ocp_cartpendulum_discrete,
)
import casadi as cas
from acados_template import (
    AcadosOcp,
    AcadosSim,
    AcadosSimSolver,
    AcadosOcpSolver,
    AcadosOcpOptions,
    ZoroDescription,
)

N_horizon = 80  # number of steps in the horizon
Tf = 2  # time horizon [s]
N_sim = 250   # numer of simulation steps
Ts_st = Tf / N_horizon

# Definition of AcadosOcpOptions 
ocp_opts = AcadosOcpOptions()
ocp_opts.tf = Tf
ocp_opts.N_horizon = N_horizon 
ocp_opts.qp_solver = "FULL_CONDENSING_HPIPM"


nominal_model = export_pendulum_ode_model_with_discrete_rk4(Ts_st)

real_model = export_pendulum_ode_real_model_with_discrete_rk4(Ts_st)

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
next_pred_state = np.array(next_pred_state)   # [p, theta, v, omega]

# GENERATE THE GP DATA
# Input to GP
U_sim_padded = np.vstack([U_sim, np.zeros((1, 1))])
X_gp = np.hstack([X_sim, U_sim_padded])

# print("X_gp shape is ", X_gp.shape)
# print("next_pred_state shape is ", next_pred_state.shape)

# Targets 
# Grey Box model: target is the difference between the next predicted state and the next real state   
# Y_gp_v = X_gp[1:, 2] - next_pred_state[:, 2]
# Y_gp_w = X_gp[1:, 3] - next_pred_state[:, 3]
# Black Box model: target is the difference between the next and the actual state
Y_gp_v = X_gp[1:, 1] - X_gp[:-1, 1]
Y_gp_w = X_gp[1:, 3] - X_gp[:-1, 3]

# === Training Phase (use real system simulation as before) ===
# Assume X_gp and Y_gp are filled from first simulation

# Train the GP model on full dynamics (not residuals)
gp_input = torch.tensor(X_gp[:-1, :], dtype=torch.float32)
gp_output = torch.tensor(np.hstack([Y_gp_v.reshape(-1, 1), Y_gp_w.reshape(-1, 1)]), dtype=torch.float32)
likelihood = MultitaskGaussianLikelihood(num_tasks=4)
gp_model = BatchIndependentMultitaskGPModel(
    train_x=gp_input,
    train_y=gp_output,
    input_dimension=5,
    residual_dimension=4,
    likelihood=likelihood,
)
gp_model, likelihood = train_gp_model(gp_model, training_iterations=300)

# Extract GP model data to export to C
tx, alpha, ls, os = extract_model_params(gp_model)
export_gp_to_c(tx, alpha, ls, os, file_path="gp_dynamics.c")

# Compile gp_dynamics.c manually or with CMake to produce libgp_dynamics.so

# === MPC with compiled GP model ===

# Load the compiled C dynamics
lib = ctypes.CDLL("./build/libgp_dynamics.so")  # path to compiled .so

dt = 0.025
N = 80
nx = 4
nu = 1
Tf = dt * N

# Create acados ocp
ocp_config = ocp_nlp_config_create("full_condensing_hpipm")
ocp_dims = ocp_nlp_dims_create(ocp_config)
ocp_dims_set(ocp_dims, "nx", nx)
ocp_dims_set(ocp_dims, "nu", nu)
ocp_dims_set(ocp_dims, "ny", nx + nu)
ocp_dims_set(ocp_dims, "ny_e", nx)
ocp_dims_set(ocp_dims, "N", N)

ocp_opts = ocp_nlp_opts_create(ocp_config, ocp_dims)
ocp_nlp_opts_set(ocp_config, ocp_opts, "integrator_type", "DISCRETE")
ocp_nlp_opts_set(ocp_config, ocp_opts, "nlp_solver", "SQP")
ocp_nlp_opts_set(ocp_config, ocp_opts, "qp_solver", "FULL_CONDENSING_HPIPM")
ocp_nlp_opts_set(ocp_config, ocp_opts, "tf", Tf)

# External function pointer setup
ext_fun = external_function_generic()
ext_fun.casadi_fun = lib.gp_dynamics
ext_fun.casadi_work = None
ext_fun.casadi_sparsity_in = None
ext_fun.casadi_sparsity_out = None
ext_fun.casadi_n_in = 1
ext_fun.casadi_n_out = 1

# Create and populate ocp_nlp_in
ocp_nlp_in = ocp_nlp_in_create(ocp_config, ocp_dims)

# Set cost: simple quadratic cost on x and u
W = np.eye(nx + nu)
W_e = np.eye(nx)
yref = np.zeros(nx + nu)
yref_e = np.zeros(nx)

for stage in range(N):
    ocp_nlp_cost_model_set(ocp_config, ocp_dims, ocp_nlp_in, stage, "W", W)
    ocp_nlp_cost_model_set(ocp_config, ocp_dims, ocp_nlp_in, stage, "yref", yref)
for stage in range(N+1):
    ocp_nlp_dynamics_model_set(ocp_config, ocp_dims, ocp_nlp_in, stage, "discrete_model", ext_fun)
ocp_nlp_cost_model_set(ocp_config, ocp_dims, ocp_nlp_in, N, "W", W_e)
ocp_nlp_cost_model_set(ocp_config, ocp_dims, ocp_nlp_in, N, "yref", yref_e)

# Set bounds and initial state
lbu = np.array([-50.0])
ubu = np.array([50.0])
idxbu = np.array([0])

for stage in range(N):
    ocp_nlp_constraints_model_set(ocp_config, ocp_dims, ocp_nlp_in, stage, "lbu", lbu)
    ocp_nlp_constraints_model_set(ocp_config, ocp_dims, ocp_nlp_in, stage, "ubu", ubu)
    ocp_nlp_constraints_model_set(ocp_config, ocp_dims, ocp_nlp_in, stage, "idxbu", idxbu)

x0 = np.array([0.0, 0.0, np.pi, 0.0])
ocp_nlp_constraints_model_set(ocp_config, ocp_dims, ocp_nlp_in, 0, "x0", x0)

# Create solver and output
ocp_nlp_out = ocp_nlp_out_create(ocp_config, ocp_dims)
ocp_solver = ocp_nlp_solver_create(ocp_config, ocp_dims, ocp_nlp_in, ocp_opts)

# Solve the OCP
total_steps = 100
x_traj = [x0.copy()]
for i in range(total_steps):
    ocp_nlp_constraints_model_set(ocp_config, ocp_dims, ocp_nlp_in, 0, "x0", x_traj[-1])
    ocp_nlp_solver_solve(ocp_solver, ocp_nlp_in, ocp_nlp_out)
    u0 = ocp_nlp_out_get(ocp_config, ocp_dims, ocp_nlp_out, 0, "u")
    x_next = x_traj[-1] + dt * np.array([x_traj[-1][1], u0[0], x_traj[-1][3], -9.81 * x_traj[-1][2]])
    x_traj.append(x_next.copy())

x_traj = np.array(x_traj)
plt.plot(x_traj)
plt.title("State Trajectory")
plt.grid()
plt.show()
