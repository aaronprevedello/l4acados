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
    BatchIndependentInducingPointGpModel
)

from l4acados.models.pytorch_models.gpytorch_models.gpytorch_residual_model import (
    GPyTorchResidualModel,
)

# Dataset and model parameters paths
dataset_path = "nominal.csv"
checkpoint_path = "gp_model.pth"

use_simsolver = True # if True use the original AcadosSimSolver, otherwise use the manual discrete integrator
real_model_type = "no_delay" # no_delay, act_delay, "double_delay"

train_flag = False
# Parameters of the cart-pendulum model
param_tau = 0.02  
param_l_nom = 0.5
param_l_real = 0.5

# SIMULATION PARAMETERS
ts_real = 0.01 # [s] real simulation integration step
integration_steps_ratio = 1 # ratio between ts_real and ts_mpc
ts_mpc = integration_steps_ratio * ts_real # mpc integration step
time_horizon_mpc = 2 # [s] mpc time horizon
n_steps_mpc_horizon = int(time_horizon_mpc / ts_mpc) # number of steps in the mpc prediction horizon

simulation_time = 6 # [s] duration of the real simulation
sim_steps = int(simulation_time / ts_real) # number of steps in the simulation

# Define mapping between GP outputs and model states 
B_m = np.array([
    [0, 0.0],   # ts_mpc/2
    [0.0, 0],   # ts_mpc/2
    [1.0, 0.0],
    [0.0, 1.0]
])

# Cost matrices for the MPC controller
Q = np.diagflat([10.0, 10.0, 0.001, 0.001])  # [cart, theta, cart_vel, omega]
R = np.array([[0.01]])                    # [u]
Qe = np.diagflat([10.0, 10.0, 0.001, 0.001])  # terminal cost

#Q = np.diagflat([15.0, 10.0, 0.9, 0.1])  # [cart, theta, cart_vel, omega]
#R = np.array([0.1])                    # [u]
#Qe = 10*np.diagflat([10.0, 10.0, 0.1, 0.1])  # terminal cost

# Dictionary of the simulation setup parameters
setup_dict = {
    "param_tau" : param_tau,
    "param_l_nom" : param_l_nom,
    "param_l_real" : param_l_real,
    "ts_real" : ts_real,
    "integration_steps_ratio" : integration_steps_ratio, 
    "ts_mpc" : ts_mpc,
    "time_horizon_mpc" : time_horizon_mpc,
    "n_steps_mpc_horizon" : n_steps_mpc_horizon,
    "B_m" : B_m,
    "Q" : Q, 
    "R" : R, 
    "Qe" : Qe,    
}

delete_json_files()

# Define nominal ocp model that will be used in MPC
ocp = export_ocp_cartpendulum(n_steps_mpc_horizon, time_horizon_mpc, export_discrete_pendulum_ode_model(ts_mpc, black_box=True), "ERK", Q, R, Qe)
ocp.parameter_values = np.array([param_l_nom])
ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp_no_dynamics.json')

# Loading the GP datasets
print("Loading GP datasets...")
n_gp_outputs = 2
train_inputs, train_outputs = load_gp_data_from_csv(dataset_path, n_gp_outputs)
print("train inputs shape is: ", train_inputs.shape)
print("train output shape is: ", train_outputs.shape)

# Loading the GP model
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks = n_gp_outputs)
gp_model = BatchIndependentMultitaskGPModel(
    train_x=train_inputs,
    train_y=train_outputs, 
    input_dimension=ocp.dims.nx + ocp.dims.nu, # should be 4+1=5
    residual_dimension=n_gp_outputs, 
    likelihood=likelihood, 
    use_ard=True
)

# Load State dicts if the model is already trained
checkpoint = torch.load(checkpoint_path, weights_only=True)
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

# Defint the residual GP model
residual_model = GPyTorchResidualModel(gp_model)

# Define l4acados solver object
l4acados_solver = l4acados.controllers.ResidualLearningMPC(
    ocp=ocp, 
    B=B_m, 
    residual_model=residual_model,
    use_ocp_model=True, 
    use_cython=False,
    path_json_ocp='residual_ocp.json',
    path_json_sim='residual_sim.json',
    use_simsolver=use_simsolver,
)

# Setup of the AcadosSimSolver object
real_model = export_pendulum_ode_model(ts_real) # standard model that is simulated, absence of disturbances
sim = AcadosSim()
sim.model = real_model  # real model with full dynamics
sim.parameter_values = np.array([param_l_real])
sim.solver_options.integrator_type = "ERK"
sim.solver_options.T = ts_real
sim_solver = AcadosSimSolver(sim, json_file = "acados_sim_nominal.json")
sim_solver.model = real_model
sim_solver.set("p", np.array([param_l_real]))

# Initial state of the simulation
x0 = np.array([0, np.pi, 0.0, 0.0])
x_current = x0.copy()
pred_state = np.array([0, 0, 0, 0])

# Define lists where to store simulation data
X_sim = [x0.copy()]
U_sim = []
next_pred_state = []
cpt_time = []

# Empty the folder where live data are stored, useful for simulation live visualization and for simulations that fail
save_dir = "grafici"
for file in os.listdir(save_dir):
    file_path = os.path.join(save_dir, file)
    if os.path.isfile(file_path):
        os.remove(file_path)

# Define the arrays where to store the predicted state's trajectories
p_predicted = np.zeros((sim_steps, n_steps_mpc_horizon))
theta_predicted = np.zeros((sim_steps, n_steps_mpc_horizon))
v_predicted = np.zeros((sim_steps, n_steps_mpc_horizon))
omega_predicted = np.zeros((sim_steps, n_steps_mpc_horizon))

# Initialize solver (don't know if it is useful)
for stage in range(n_steps_mpc_horizon):
    ocp_solver.set(stage, "x", x0)
    ocp_solver.set(stage, "u", np.zeros((1,)))

    ocp_solver.set(n_steps_mpc_horizon, "x", x0)

print("model name: ", sim_solver.model.name)

print("Starting simulation...")

# Simulation loop
for i in range(sim_steps):

    print("---------------------------------------------------\n\n\n")
    print(f"Iteration: {i+1}/{sim_steps}, time: {i*ts_real:.3f}s")
    print("Actual state is ", x_current)
    print("Prediction error on the next state is ", x_current[0:4] - pred_state)

    # Set the curreent state in OCP solver
    l4acados_solver.set(0, "lbx", x_current[0:4])
    l4acados_solver.set(0, "ubx", x_current[0:4])

    # Solve the OCP problem
    l4acados_solver.solve()

    # Print ocp solver statistics
    print(
      f"CPT: {l4acados_solver.ocp_solver.get_stats('time_tot')*1000:.2f}ms |\n "
      f"Shooting (linearization): {l4acados_solver.ocp_solver.get_stats('time_lin')*1000:.2f}ms |\n "
      f"QP Solve: {l4acados_solver.ocp_solver.get_stats('time_qp_solver_call')*1000:.2f}ms |\n "
      f"Opt. Crit: {l4acados_solver.ocp_solver.get_stats('residuals')[0]:.3e} |\n "
      f"SQP Iter: {l4acados_solver.ocp_solver.get_stats('sqp_iter')}")

    cpt_time.append(l4acados_solver.ocp_solver.get_stats('time_tot')*1000)

    # Get optimal control input
    u0 = l4acados_solver.get(0, "u")
    print("Computed optimal control input u0: ", u0)

    # Get next predicted state from the ocp problem
    pred_state = l4acados_solver.get(1, "x")
    print("next predicted state is: ", pred_state)

    # Store the whole prediction horizon
    for j in range(l4acados_solver.N):   
        p_predicted[i, j] = l4acados_solver.get(j, "x")[0]
        theta_predicted[i, j] = l4acados_solver.get(j, "x")[1]
        v_predicted[i, j] = l4acados_solver.get(j, "x")[2]
        omega_predicted[i, j] = l4acados_solver.get(j, "x")[3]

    # Simulate the real system with the AcadosSimSolver
    print("Simulating with AcadosSimSolver")
    sim_solver.set("x", x_current)
    sim_solver.set("u", u0)

    print("SIM SOLVER ACTUAL STATE: ", sim_solver.get("x"))
    #print("SIM SOLVER ACTUAL INPUT: ", sim_solver.get("u"))

    # now solve
    status = sim_solver.solve()
    if status != 0:
        print(f"WARNING: sim_solver.solve() returned nonzero status: {status}")


    # Catch next simulated state
    x_next = sim_solver.get("x")
    print("Next state from the simulation is: ", x_next)

    # Update the current state 
    x_current = x_next.copy()

    # Update the lists
    X_sim.append(x_current.copy())
    U_sim.append(u0)

    # Generate data for live plotting
    data_dict = {
            "i": i*ts_real,
            "x": x_current.copy(),
            "u": u0.copy(),
            "p_predicted": p_predicted[i, :].copy(),
            "theta_predicted": theta_predicted[i,:].copy(),
            "v_predicted": v_predicted[i,:].copy(),
            "omega_predicted": omega_predicted[i,:].copy(),
        }
    # Save data in the directory
    np.savez(os.path.join(save_dir, f"step_{i:03d}.npz"), **data_dict)

print("Simulation completed!")

# Transform lists to numpy arrays
X_parallel = np.array(X_sim[:-1])
U_parallel = np.array(U_sim)
cpt_time = np.array(cpt_time)

np.savez("control_simulation.npz", X_sim=X_parallel, U_sim=U_parallel, setup_dict=setup_dict)
print("Simulation data saved to controlsimulation.npz")

# Plot trajectories
# Time vector of the simulation for plotting
time_vec = np.linspace(0, simulation_time, sim_steps)
# Time vector for MPC prediction horizon
t_horizon = np.linspace(0, time_horizon_mpc, n_steps_mpc_horizon)
# Plot the results
plt.figure(figsize=(10, 6))
# First subplot
plt.subplot(4, 1, 1)
#plt.plot(time_vec, X_sim[:, 0], label='Cart Position') 
#plt.plot(time_vec, X_sim[:, 1], label='Pendulum Angle')
plt.plot(time_vec, X_parallel[:, 0], 'orange', label='Discrete cart position')
# add predicted horizons every 10 steps
for i in range(0, sim_steps, 10):
    plt.plot(i*ts_real + t_horizon, p_predicted[i, :], 'r--', alpha=0.5, label='Predicted Cart' if i==0 else "")
plt.xlabel('time (s)')
plt.ylabel('angle (rad) / position (m)')
plt.legend()
plt.grid()

# Second subplot
plt.subplot(4, 1, 2)
#plt.plot(time_vec, X_sim[:, 0], label='Cart Position') 
#plt.plot(time_vec, X_sim[:, 1], label='Pendulum Angle')
plt.plot(time_vec, X_parallel[:, 1], 'lime', label='Discrete pendulum angle')
# add predicted horizons every 10 steps
for i in range(0, sim_steps, 10):
    plt.plot(i*ts_real + t_horizon, theta_predicted[i, :], 'g--', alpha=0.5, label='Predicted Theta' if i==0 else "")
plt.xlabel('time (s)')
plt.ylabel('angle (rad) / position (m)')
plt.legend()
plt.grid()

# Third subplot
plt.subplot(4, 1, 3)
#plt.plot(time_vec, U_sim, label='Control Input (Force)')
plt.plot(time_vec, U_parallel, label='Discrete Integrator Force')
if X_parallel.shape[1] >= 5:
    plt.plot(time_vec, X_parallel[:, 4], label='actual input')
if X_parallel.shape[1] >= 6:
    plt.plot(time_vec, X_parallel[:, 5], label='past input')
plt.grid()
plt.xlabel('time (s)')
plt.ylabel('force (N)')
plt.legend()

# Fourth subplot
plt.subplot(4, 1, 4)
plt.plot(time_vec, X_parallel[:, 2], label='cart velocity')
plt.plot(time_vec, X_parallel[:, 3], label='pendulum velocity')
plt.grid()
plt.xlabel('time (s)')
plt.ylabel('velocity (m/s, rad/s)')
plt.legend()
plt.savefig("control_simulation.svg")

plt.figure()
plt.plot(time_vec, cpt_time, label='CPT time')
plt.grid()
plt.xlabel('time (s)')
plt.ylabel('time (ms)')
plt.legend()
plt.show()

print("Figure saved as control_simulation.svg")
# visualize_inverted_pendulum(X_sim, U_sim, time_vec)
visualize_inverted_pendulum(X_parallel, U_parallel, time_vec, ts_real)


