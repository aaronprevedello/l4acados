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

time_ref, p_ref, theta_ref, v_ref, omega_ref = vertical_points_ref(Ts_st, N_sim, N_horizon)
# time_ref, p_ref, theta_ref, v_ref, omega_ref = sinusoidal_ref(Ts_st, N_sim, N_horizon)
# plot_references(time_ref, p_ref, theta_ref, v_ref, omega_ref)

# Definition of AcadosOcpOptions 
ocp_opts = AcadosOcpOptions()
ocp_opts.tf = Tf
ocp_opts.N_horizon = N_horizon 
ocp_opts.qp_solver = "FULL_CONDENSING_HPIPM"

nominal_model = export_my_augmented_pendulum_ode_model_with_discrete_rk4(Ts_st, black_box=False) # with black_box false w do not disable the dynamics for v and omega
# print("nominal_model.p is ", nominal_model.p)

real_model = export_augmented_pendulum_ode_model(Ts_st)
# real_model.set("p", 0.5)
# print("real_model.p is ", real_model.p)

ocp = export_augmented_ocp_cartpendulum_discrete(N_horizon, Tf, nominal_model)   
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

    # Solve the OCP 
    ocp_solver.solve()

    # Get the optimal control input and the predicted state
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
next_pred_state = np.array(next_pred_state)   # [p, theta, v, omega and all the past states]

# GENERATE THE GP DATA
# Input to GP
X_gp = np.hstack([X_sim[:-1, :], U_sim])

# print("X_gp shape is ", X_gp.shape)
# print("next_pred_state shape is ", next_pred_state.shape)

# Targets 
# Grey Box model: target is the difference between the next predicted state and the next real state   
# Y_gp_v = X_gp[1:, 2] - next_pred_state[:, 2]
# Y_gp_w = X_gp[1:, 3] - next_pred_state[:, 3]

# Black Box model: target is the difference between the next and the actual state
Y_gp_v = X_gp[1:, 2] - X_gp[:-1, 2]
Y_gp_w = X_gp[1:, 3] - X_gp[:-1, 3]

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
plt.plot(time_ref, p_ref, label='Cart Ref')
plt.plot(time_ref, theta_ref, label='Pendulum Ref')
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

# Define the GP model
train_inputs = torch.tensor(X_gp[:-1, :], dtype=torch.float32)  
train_outputs = torch.tensor(np.hstack([Y_gp_v.reshape(-1, 1), Y_gp_w.reshape(-1, 1)]), dtype=torch.float32)
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks = 2)
gp_model = BatchIndependentMultitaskGPModel(
    train_x = train_inputs,
    train_y = train_outputs,
    input_dimension=13,
    residual_dimension=2,
    likelihood=likelihood,
)

# Train the GP model on the data   
gp_model, likelihood = train_gp_model(
    gp_model, training_iterations=500, learning_rate=0.3)


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

# Plot the GP model fit on training data
#plot_gp_fit_on_training_data(
#    train_inputs,
#    train_outputs,
#    gp_model,
#    likelihood,
#)

# Define residual model
res_model = GPyTorchResidualModel(gp_model)

# Define mapping from gp outputs (dim=2) to model states (dim=4)
B_m = np.array([
    [Ts_st / 2, 0],
    [0, Ts_st / 2],
    [1.0, 0],
    [0, 1.0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
])

# New simulation using the residual model
nominal_model = export_my_augmented_pendulum_ode_model_with_discrete_rk4(Ts_st, black_box=True)

real_model = export_my_augmented_pendulum_ode_model_with_discrete_rk4(Ts_st)

ocp = export_augmented_ocp_cartpendulum_discrete(N_horizon, Tf, nominal_model)   
ocp.parameter_values = np.array([0.8])
ocp.solver_options.integrator_type = "IRK"  # valid types: "ERK", "IRK", "GNSF"
ocp.solver_options.nlp_solver_tol_eq = 1e-2
ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")

# Define l4acados solver
l4acados_solver = ResidualLearningMPC(
    ocp=ocp,
    residual_model=res_model,
    B = B_m,
)

sim = AcadosSim()
sim.model = real_model
sim.parameter_values = np.array([0.5])
sim.solver_options.integrator_type = "ERK"  # valid types: "ERK", "IRK", "GNSF"
sim.solver_options.T = Tf / N_horizon

sim_solver = AcadosSimSolver(sim, json_file="acados_sim.json")
sim_solver.model = real_model
sim_solver.set("p", 0.5)

x0 = np.concatenate((np.array([0.0, np.pi, 0.0, 0.0]), np.zeros((8, ))))  # start slightly off upright
# x0 = np.tile(np.array([0.0, np.pi, 0.0, 0.0]), 3)
x_current = x0.copy()

X_sim_res = [x0.copy()]
U_sim_res = []
X_gp = np.array([]) # composed by all the states and the input
Y_gp = np.array([]) # difference between next predicted state and the next real state


save_dir = "grafici"
for file in os.listdir(save_dir):
    file_path = os.path.join(save_dir, file)
    if os.path.isfile(file_path):
        os.remove(file_path)


print(f"Prediction horizon is {Tf} [s]")
print(f"Number of steps in the horizon is {N_horizon}")
print(f"Integration step is {Ts_st} [s]")
# Simulate the system with the residual model
for i in range(N_sim):
    # Computation of the optimal control input with nominal model + residual model
    print("Second simulation iteration ", i)
    l4acados_solver.set(0, "lbx", x_current)
    l4acados_solver.set(0, "ubx", x_current)

    # Update cost reference at each stage in the horizon
    for stage in range(l4acados_solver.N):
        stage_yref = np.array([p_ref[i+stage], theta_ref[i+stage], v_ref[i+stage], omega_ref[i+stage],0, 0, 0, 0, 0, 0, 0, 0, 0])
        l4acados_solver.set(stage, "yref", stage_yref)
    # Optionally also set terminal cost reference (if used)
    l4acados_solver.set(l4acados_solver.N, "yref", np.array([p_ref[i+l4acados_solver.N], theta_ref[i+l4acados_solver.N], v_ref[i+l4acados_solver.N], omega_ref[i+l4acados_solver.N], 0, 0, 0, 0, 0, 0, 0, 0]))



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
    
    # Generate data to be saved
    data_dict = {
            "i": i*Ts_st,
            "x": x_current.copy(),
            "u": u0.copy(),
        }
    # Save data in the directory
    np.savez(os.path.join(save_dir, f"step_{i:03d}.npz"), **data_dict)

    
    # Update current state
    X_sim_res.append(x_next.copy())
    x_current = x_next.copy()

    

print("End of the simulation")


# Convert lists to numpy arrays
X_sim_res = np.array(X_sim_res)
U_sim_res = np.array(U_sim_res)


# Time vector
time_vec = np.linspace(0, Tf / N_horizon * N_sim, N_sim + 1)

# Plot the results
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(time_vec, X_sim_res[:, 0], '+', label='Res Cart Position')
plt.plot(time_vec, X_sim[:, 0], '+', label='Cart Position')
plt.xlabel('time (s)')
plt.ylabel('position (m)')
plt.legend()
plt.grid()
plt.subplot(3, 1, 2)
plt.plot(time_vec[:-1], U_sim_res, '+', label='Res Control Input (Force)')
plt.plot(time_vec[:-1], U_sim, '+', label='Control Input (Force)')
plt.grid()
plt.xlabel('time (s)')
plt.ylabel('force (N)')
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(time_vec, X_sim_res[:, 2], '+', label='Res Pendulum Angle')
plt.plot(time_vec, X_sim[:, 2], '+', label='Pendulum Angle')
plt.ylabel('angle (rad)')
plt.xlabel('time (s)')
plt.legend()
plt.grid()
plt.show()

np.savez("rollout_data_res_ctrl.npz", X_sim_res=X_sim_res, U_sim_res=U_sim_res)