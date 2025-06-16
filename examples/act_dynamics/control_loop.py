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
ref_type = 'swing_up' # 'swing_up', 'vertical', 'sin', 'mix', 'rich_mix'
param_tau = 0.05
param_l_nom = 0.5
param_l_real = 0.5

Ts_real = 0.025 # [s] real model integration step
integration_steps_ratio = 1 # ratio between Ts_st and T_real
Ts_st = integration_steps_ratio * Ts_real # integrator time step
Tf = 1.5  # time horizon of the mpc prediction [s]

N_horizon = int(Tf / Ts_st) # number of steps in the horizon

T_sim = 5 # [s]
sim_steps = int(T_sim/Ts_real)  # number of simulation steps

if ref_type == 'vertical':
    time_ref, p_ref, theta_ref, v_ref, omega_ref = vertical_points_ref(Ts_st, sim_steps, N_horizon*integration_steps_ratio)
elif ref_type == 'sin':    
    time_ref, p_ref, theta_ref, v_ref, omega_ref = sinusoidal_ref(Ts_st, sim_steps, N_horizon*integration_steps_ratio)
elif ref_type == 'mix':
    time_ref, p_ref, theta_ref, v_ref, omega_ref = mix_ref(Ts_st, sim_steps, N_horizon*integration_steps_ratio)
elif ref_type == 'rich_mix':
    time_ref, p_ref, theta_ref, v_ref, omega_ref = rich_mix_ref(Ts_st, sim_steps, N_horizon*integration_steps_ratio)
# plot_references(time_ref, p_ref, theta_ref, v_ref, omega_ref)

# New simulation using the residual model
nominal_model = export_pendulum_ode_model_with_discrete_rk4(Ts_st, black_box=True)

real_model = export_pendulum_ode_model(black_box=False)

# Definition of AcadosOcpOptions
ocp_opts = AcadosOcpOptions()
ocp_opts.tf = Tf
ocp_opts.N_horizons = N_horizon 
ocp_opts.Tsim = Ts_st
ocp_opts.qp_solver = "FULL_CONDENSING_HPIPM"

ocp = export_ocp_cartpendulum_discrete(N_horizon, Tf, nominal_model, integrator_type="DISCRETE")   
ocp.parameter_values = np.array([param_l_nom])
ocp.solver_options = ocp_opts
ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")

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
#print("Fin qui")
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

# Define residual model
res_model = GPyTorchResidualModel(gp_model)
print("GPyTorchResidualModel defined")

# Define mapping from gp outputs (dim=4) to model states (dim=4)
B_m = np.array([
    [0, 0],# Ts_st / 2 in third column
    [0, 0],# Ts_st / 2 in fourth column
    [1, 0],
    [0, 1]
])

B_m_augmented = np.array([
    [1, 0, 0, 0],  # Ts_st / 2 in third column
    [0, 1, 0, 0],  # Ts_st / 2 in fourth column
    [0, 0, 1.0, 0],
    [0, 0, 0, 1.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0]
])

#new_ocp = export_ocp_cartpendulum_discrete(N_horizon, Tf, nominal_model, integrator_type="DISCRETE")
#new_ocp.parameter_values = np.array([param_l_nom])
#new_ocp.solver_options = ocp_opts
#nominal_ocp_solver = AcadosOcpSolver(new_ocp, json_file="my_acados_ocp.json")

# Define l4acados solver
l4acados_solver = ResidualLearningMPC(   # try to set use_cython=False
    ocp=ocp,
    residual_model=res_model,   
    B = B_m
)
print("ResidualModelMPC defined")
# Initial condition
x0 = np.array([1.0, np.pi, 0.0, 0.0])  
x_current = x0.copy()

sim = AcadosSim()
sim.model = real_model
sim.parameter_values = np.array([param_l_real])
sim.solver_options.integrator_type = "ERK"  # valid types: "ERK", "IRK", "GNSF"
sim.solver_options.T = Ts_real

sim_solver = AcadosSimSolver(sim, json_file="acados_sim.json")
sim_solver.model = real_model
sim_solver.set("p", np.array([param_l_real]))
sim_solver.set("x", x0)

X_nom_model = [x0.copy()]
X_real = [x0.copy()]
U_des = []
X_gp = np.array([]) # composed by all the states and the input + previous positions of pendulum and cart
Y_gp = np.array([]) # difference between next predicted state and the next real state

# Empty the folder where live data are stored
save_dir = "grafici"
for file in os.listdir(save_dir):
    file_path = os.path.join(save_dir, file)
    if os.path.isfile(file_path):
        os.remove(file_path)

#Define the arrays where to store the predicted state's trajectories
#p_predicted = np.zeros((N_ocp, N_horizon))
#theta_predicted = np.zeros((N_ocp, N_horizon))
#v_predicted = np.zeros((N_ocp, N_horizon))
#omega_predicted = np.zeros((N_ocp, N_horizon))

# Simulate the system with the residual model
for i in range(sim_steps):
    # Computation of the optimal control input with nominal model + residual model
    print("Second simulation iteration ", i)

    # Build augmented nominal state (exclude u_act)
    #x_aug = build_augmented_state(state_history, input_history)
    #if i % steps_per_mpc == 0:
    print("actual state is ", x_current)
    l4acados_solver.ocp_solver.set(0, "lbx", x_current)
    l4acados_solver.ocp_solver.set(0, "ubx", x_current)
    l4acados_solver.ocp_solver.set(0, "x", x_current)
    # Update cost reference at each stage in the horizon
    if ref_type != 'swing_up':
        for stage in range(l4acados_solver.N):
            stage_yref = np.array([p_ref[i+stage], theta_ref[i+stage], v_ref[i+stage], omega_ref[i+stage],0])
            l4acados_solver.ocp_solver.set(stage, "yref", stage_yref)
        # Optionally also set terminal cost reference (if used)
        l4acados_solver.ocp_solver.set(l4acados_solver.N, "yref", np.array([p_ref[i+l4acados_solver.N], theta_ref[i+l4acados_solver.N], v_ref[i+l4acados_solver.N], omega_ref[i+l4acados_solver.N]]))

    l4acados_solver.ocp_solver.solve()

    # Print ocp solver statistics
    print(
      f"CPT: {ocp_solver.get_stats('time_tot')*1000:.2f}ms |\n "
      f"Shooting (linearization): {ocp_solver.get_stats('time_lin')*1000:.2f}ms |\n "
      f"QP Solve: {ocp_solver.get_stats('time_qp_solver_call')*1000:.2f}ms |\n "
      f"Opt. Crit: {ocp_solver.get_stats('residuals')[0]:.3e} |\n "
      f"Statistics: {ocp_solver.get_stats('statistics')} | \n"
      f"SQP Iter: {ocp_solver.get_stats('sqp_iter')}")

    u0 = l4acados_solver.ocp_solver.get(0, "u")
    print("u0      = ", u0)
    #residual_prediction = res_model.evaluate([x_current, u0])
    #print("residual prediction is ", residual_prediction)
    pred_state = l4acados_solver.ocp_solver.get(1, "x")
    print("next predicted state is ", pred_state)
    # Store the whole prediction horizon 
    #for j in range(l4acados_solver.N):   
    #    p_predicted[i, j] = l4acados_solver.get(j, "x")[0]
    #    theta_predicted[i, j] = l4acados_solver.get(j, "x")[1]
    #    v_predicted[i, j] = l4acados_solver.get(j, "x")[2]
    #    omega_predicted[i, j] = l4acados_solver.get(j, "x")[3]
    # print("u0 is ", u0.shape)
    # print("U_sim is ", U_sim.shape)

    U_des.append(u0)

    # Simulation of the real model
    sim_solver.set("x", x_current)
    sim_solver.set("u", u0)
    sim_solver.solve()

    # Get the next state from the simulation
    x_next = sim_solver.get("x")
    X_real.append(x_next.copy())

    # Update buffers (exclude u_act for controller's history)
    #state_history.append(x_next[0:4].copy())
    #input_history.append(u0.copy())
    x_current = x_next.copy() 
    # Generate data to be saved
    data_dict = {
            "i": i*Ts_st,
            "x": x_current.copy(),
            "u": u0.copy(),
        }
    # Save data in the directory
    np.savez(os.path.join(save_dir, f"step_{i:03d}.npz"), **data_dict)

    print("###############################################################\n\n\n")
    # Update current state
    # Augment x_next before appending
    #augmented_x_next = np.concatenate((
    #    x_next[0:4],                          # current state
    #    state_history[-2][0:2].copy(),            # previous state
    #    input_history[-2].copy(),            # previous input
    #    state_history[-3][0:2].copy(),            # two steps ago
    #    input_history[-3].copy()             # two steps ago
    #))
    #X_nom_model.append(augmented_x_next.copy())
       

print("End of the simulation")

