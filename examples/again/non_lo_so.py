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

train_flag = False
param_tau = 0.02    
param_l_nom = 0.5
param_l_real = 0.5

ts_real = 0.025 # [s] real simulation integration step
integration_steps_ratio = 1 # ratio between ts_real and ts_mpc
ts_mpc = integration_steps_ratio * ts_real # mpc integration step
time_horizon_mpc = 1.5 # [s] mpc time horizon
n_steps_mpc_horizon = int(time_horizon_mpc / ts_mpc) # number of steps in the mpc prediction horizon

simulation_time = 10 # [s] duration of the real simulation
sim_steps = int(simulation_time / ts_real) # number of steps in the simulation

# Define mapping between GP outputs and model states 
B_m = np.array([
    [ts_mpc/2, 0.0],   # ts_mpc/2
    [0.0, ts_mpc/2],   # ts_mpc/2
    [1.0, 0.0],
    [0.0, 1.0]
])

Q = np.diagflat([15.0, 10.0, 0.9, 0.1])  # [cart, theta, cart_vel, omega]
R = np.array([[0.5]])                    # [u]
Qe = 5*np.diagflat([10.0, 10.0, 0.1, 0.1])  # terminal cost

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

# Nominal ocp model
ocp = export_ocp_cartpendulum(n_steps_mpc_horizon, time_horizon_mpc, export_discrete_pendulum_ode_model(ts_mpc), "ERK", Q, R, Qe)
ocp.parameter_values = np.array([param_l_nom])
ocp_solver  = AcadosOcpSolver(ocp, json_file='acados_ocp.json')

# Loading of gp datasets
print("loading GP datasets")
n_gp_outputs = 2
train_inputs, train_outputs = load_gp_data_from_csv("X_sim_with_gp_targets.csv", n_gp_outputs)
# train_inputs = torch.tensor(train_inputs, dtype=torch.float32) 
# train_outputs = torch.tensor(train_outputs, dtype = torch.float32)
print("train inputs shape is ", train_inputs.shape)
print("train_outputs shape is ", train_outputs.shape)

# Loading the GP model
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks = n_gp_outputs)
gp_model = BatchIndependentMultitaskGPModel(
    train_x = train_inputs,
    train_y = train_outputs,
    input_dimension=ocp.dims.nx + ocp.dims.nu, # 4 + 1 = 5
    residual_dimension=n_gp_outputs,
    likelihood=likelihood,
    #use_ard=True,
    #inducing_points = 100,
)

# Train the GP model on the data  
if train_flag:
    gp_model.train()
    likelihood.train()
    print("Training the GP model...")
    gp_model, likelihood = train_gp_model(
        gp_model, training_iterations=1000, learning_rate=0.1)
    print("GP model trained successfully")
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

residual_model = GPyTorchResidualModel(gp_model)

# Define l4acados solver 
l4acados_solver = l4acados.controllers.ResidualLearningMPC(
    ocp=ocp,
    B=B_m,
    residual_model=residual_model,
    use_ocp_model=True,
    #use_cython=False,
    path_json_ocp="residual_ocp.json",
    path_json_sim="residual_sim.json"
)

# Initial conditions
x0 = np.array([0.0, np.pi, 0.0, 0.0])
x_current = x0.copy()

# Simulation setup
#sim = AcadosSim()
#sim.model = export_pendulum_ode_model_double_delay(ts_real)
#sim.parameter_values = np.array([param_l_real, param_tau])
#sim.solver_options.integrator_type = "ERK"
#sim.solver_options.T = ts_real # real simulation step
#
#sim_solver = AcadosSimSolver(sim, json_file = "acados_sim.json")
#
##x0 = np.array([0, np.pi, 0, 0])
##x_current = x0.copy()
#sim_solver.set("x", x0)

X_sim = [x0.copy()]
U_sim = []
next_pred_state = []
u0 = np.zeros((ocp.dims.nu,))  # initial control input

# Definition of the manual discrete integrator
lin_disc_dyn_bar, f_disc_dyn_bar = export_discrete_integrator(export_pendulum_ode_model_double_delay(ts_real))
# f_disc_dyn_bar = cas.Function("f_disc_dyn_bar", [simulation_model.x, simulation_model.u, simulation_model.p], [simulation_model.x_next_lin]on
x0_actuation_dyn = np.array([0.0, np.pi, 0.0, 0.0, 0.0, 0.0])
X_parallel = [x0_actuation_dyn.copy()]
U_parallel = []
#print("DIscrete dynamics function: ", f_disc_dyn_bar)

#for i in range(sim_steps):
#
#    print("---------------------------------------------------\n\n\n")
#    print("Iteration: ", i)
#    print("x_current before solving OCP: ", x_current)
#
#    # Set current state in the OCP solver 
#    #l4acados_solver.set(0, "x", x_current)
#    l4acados_solver.set(0, "lbx", x_current)
#    l4acados_solver.set(0, "ubx", x_current)
#
#    l4acados_solver.solve()
#    print("OCP solved succesfully")
#
#    # Print ocp solver statistics
#    print(
#      f"CPT: {l4acados_solver.ocp_solver.get_stats('time_tot')*1000:.2f}ms |\n "
#      f"Shooting (linearization): {l4acados_solver.ocp_solver.get_stats('time_lin')*1000:.2f}ms |\n "
#      f"QP Solve: {l4acados_solver.ocp_solver.get_stats('time_qp_solver_call')*1000:.2f}ms |\n "
#      f"Opt. Crit: {l4acados_solver.ocp_solver.get_stats('residuals')[0]:.3e} |\n "
#      f"SQP Iter: {l4acados_solver.ocp_solver.get_stats('sqp_iter')}")
#
#    # Get optimal control input 
#    u0 = l4acados_solver.get(0, "u")
#    print("Computed optimal control input u0: ", u0)
#
#    pred_state = l4acados_solver.get(1, "x")
#    print("next predicted state is ", pred_state)
#
#    sim_solver.set("x", x_current)
#    sim_solver.set("u", u0)
#    sim_solver.solve()
#
#    x_next = sim_solver.get("x")
#    x_current = x_next.copy()
#
#    X_sim.append(x_next.copy())
#    U_sim.append(u0)

X_sim = np.array(X_sim[:-1])
U_sim = np.array(U_sim)

x_current = x0_actuation_dyn.copy()

# Empty the folder where live data are stored
save_dir = "grafici"
for file in os.listdir(save_dir):
    file_path = os.path.join(save_dir, file)
    if os.path.isfile(file_path):
        os.remove(file_path)

# Simulation loop
for i in range(sim_steps):

    print("---------------------------------------------------\n\n\n")
    print(f"Iteration: {i}/{sim_steps}")
    print("Actual state is ", x_current)

    l4acados_solver.set(0, "lbx", x_current[0:4])
    l4acados_solver.set(0, "ubx", x_current[0:4])

    l4acados_solver.solve()

    # Print ocp solver statistics
    print(
      f"CPT: {l4acados_solver.ocp_solver.get_stats('time_tot')*1000:.2f}ms |\n "
      f"Shooting (linearization): {l4acados_solver.ocp_solver.get_stats('time_lin')*1000:.2f}ms |\n "
      f"QP Solve: {l4acados_solver.ocp_solver.get_stats('time_qp_solver_call')*1000:.2f}ms |\n "
      f"Opt. Crit: {l4acados_solver.ocp_solver.get_stats('residuals')[0]:.3e} |\n "
      f"SQP Iter: {l4acados_solver.ocp_solver.get_stats('sqp_iter')}")

    # Get optimal control input 
    u0 = l4acados_solver.get(0, "u")
    print("Computed optimal control input u0: ", u0)

    # Get the predicted state from the ocp
    pred_state = l4acados_solver.get(1, "x")
    print("next predicted state is ", pred_state)

    # Discrete manual integrator 
    x_next = f_disc_dyn_bar(x_current, u0, np.array([param_l_real, param_tau])).full().flatten()
    x_current = x_next.copy()

    X_parallel.append(x_current.copy())
    U_parallel.append(u0)

    # Generate data for live plotting
    data_dict = {
            "i": i*ts_real,
            "x": x_current.copy(),
            "u": u0.copy(),
        }
    # Save data in the directory
    np.savez(os.path.join(save_dir, f"step_{i:03d}.npz"), **data_dict)

    #input("Press any key to continue... ")

X_parallel = np.array(X_parallel[:-1])
U_parallel = np.array(U_parallel)

np.savez("control_simulation.npz", X_sim=X_sim, U_sim=U_sim, setup_dict=setup_dict)

# Plot trajectories
# Time vector of the simulation for plotting
time_vec = np.linspace(0, simulation_time, sim_steps)
# Plot the results
plt.figure(figsize=(10, 6))
# First subplot
plt.subplot(3, 1, 1)
#plt.plot(time_vec, X_sim[:, 0], label='Cart Position') 
#plt.plot(time_vec, X_sim[:, 1], label='Pendulum Angle')
plt.plot(time_vec, X_parallel[:, 0], label='Discrete cart position')
plt.plot(time_vec, X_parallel[:, 1], label='Discrete pendulum angle')
plt.xlabel('time (s)')
plt.ylabel('angle (rad) / position (m)')
plt.legend()
plt.grid()
# Second subplot
plt.subplot(3, 1, 2)
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
# Third subplot
plt.subplot(3, 1, 3)
plt.plot(time_vec, X_parallel[:, 2], label='cart velocity')
plt.plot(time_vec, X_parallel[:, 3], label='pendulum velocity')
plt.grid()
plt.xlabel('time (s)')
plt.ylabel('velocity (m/s, rad/s)')
plt.legend()

#plt.figure(figsize=(10, 6))
#plt.plot(time_vec, X_sim[:, 1], label='Pendulum Angle')
#plt.plot(time_vec, X_parallel[:, 1], label='Discrete Pendulum Angle')
#plt.xlabel('time (s)')
#plt.ylabel('angle (rad)')
#plt.legend()
#plt.grid()

#plt.figure(figsize=(10,6))
#plt.plot(time_vec, X_sim[:, 0]-X_parallel[:, 0], label='Difference between cart positions')
#plt.plot(time_vec, X_sim[:, 1] - X_parallel[:, 1], label='Difference between pendulum angles')
#plt.xlabel('time (s)')
#plt.ylabel('position (m) / angle (rad)')
#plt.grid()
#plt.legend()
plt.savefig("control_simulation.jpg")
plt.show()

# visualize_inverted_pendulum(X_sim, U_sim, time_vec)
visualize_inverted_pendulum(X_parallel, U_parallel, time_vec)




