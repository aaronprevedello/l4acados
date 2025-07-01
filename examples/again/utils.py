import numpy as np
from numpy import linalg as npla
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

import pandas as pd
import torch, gpytorch
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors

import os, glob

# Plotting


@dataclass
class EllipsoidTubeData2D:
    center_data: np.ndarray = (None,)
    ellipsoid_data: np.ndarray = (None,)
    ellipsoid_colors: np.ndarray = None


def base_plot(lb_theta=None):
    fig, ax = plt.subplots()

    if lb_theta is not None:
        ax.axvline(lb_theta)

    return fig, ax


def add_plot_ellipse(ax, E, e0, n=50, **plot_args):
    # sample angle uniformly from [0,2pi] and length from [0,1]
    radius = 1.0
    theta_arr = np.linspace(0, 2 * np.pi, n)
    w_rad_arr = [[radius, theta] for theta in theta_arr]
    w_one_arr = np.array(
        [
            [w_rad[0] * np.cos(w_rad[1]), w_rad[0] * np.sin(w_rad[1])]
            for w_rad in w_rad_arr
        ]
    )
    w_ell = np.array([e0 + E @ w_one for w_one in w_one_arr])
    h = ax.plot(w_ell[:, 0], w_ell[:, 1], **plot_args)
    return h


def add_plot_trajectory(
    ax,
    tube_data: EllipsoidTubeData2D,
    color_fun=plt.cm.Blues,
    prob_tighten=1,
    **plot_args,
):
    n_data = tube_data.center_data.shape[0]
    evenly_spaced_interval = np.linspace(0.6, 1, n_data)
    colors = [color_fun(x) for x in evenly_spaced_interval]

    h_plot = ax.plot(
        tube_data.center_data[:, 0], tube_data.center_data[:, 1], **plot_args
    )
    # set color
    h_plot[0].set_color(colors[-1])

    for i, color in enumerate(colors):
        center_i = tube_data.center_data[i, :]
        if tube_data.ellipsoid_data is not None:
            ellipsoid_i = tube_data.ellipsoid_data[i, :, :]
            # get eigenvalues
            eig_val, eig_vec = npla.eig(ellipsoid_i)
            ellipsoid_i_sqrt = (
                prob_tighten
                * eig_vec
                @ np.diag(np.sqrt(np.maximum(eig_val, 0)))
                @ np.transpose(eig_vec)
            )
            # print(i, eig_val, ellipsoid_i)
            h_ell = add_plot_ellipse(ax, ellipsoid_i_sqrt, center_i, **plot_args)
            h_ell[0].set_color(color)


# OCP stuff
def get_solution(ocp_solver, x0):
    N = ocp_solver.acados_ocp.dims.N

    # get initial values
    X = np.zeros((N + 1, ocp_solver.acados_ocp.dims.nx))
    U = np.zeros((N, ocp_solver.acados_ocp.dims.nu))

    # xcurrent = x0
    X[0, :] = x0

    # solve
    status = ocp_solver.solve()

    if status != 0:
        raise Exception("acados ocp_solver returned status {}. Exiting.".format(status))

    # get data
    for i in range(N):
        X[i, :] = ocp_solver.get(i, "x")
        U[i, :] = ocp_solver.get(i, "u")

    X[N, :] = ocp_solver.get(N, "x")
    return X, U


def simulate_solution(sim_solver, x0, N, nx, nu, U):
    # get initial values
    X = np.zeros((N + 1, nx))

    # xcurrent = x0
    X[0, :] = x0

    # simulate
    for i in range(N):
        sim_solver.set("x", X[i, :])
        sim_solver.set("u", U[i, :])
        status = sim_solver.solve()
        if status != 0:
            raise Exception(
                "acados sim_solver returned status {}. Exiting.".format(status)
            )
        X[i + 1, :] = sim_solver.get("x")

    return X


def init_ocp_solver(ocp_solver, X, U):
    # initialize with nominal solution
    N = ocp_solver.acados_ocp.dims.N
    print(f"N = {N}, size_X = {X.shape}")
    for i in range(N):
        ocp_solver.set(i, "x", X[i, :])
        ocp_solver.set(i, "u", U[i, :])
    ocp_solver.set(N, "x", X[N, :])


# GP model


def get_gp_model(ocp_solver, sim_solver, sim_solver_actual, x0, Sigma_W):
    random_seed = 123
    N_sim_per_x0 = 1
    N_x0 = 10
    x0_rand_scale = 0.1

    x_train, x0_arr = generate_train_inputs_acados(
        ocp_solver,
        x0,
        N_sim_per_x0,
        N_x0,
        random_seed=random_seed,
        x0_rand_scale=x0_rand_scale,
    )

    y_train = generate_train_outputs_at_inputs(
        x_train, sim_solver, sim_solver_actual, Sigma_W
    )


def extract_model_params(model):

    model.eval()
    train_x = model.train_inputs[0].detach().cpu().numpy()    # (N, D)
    train_y = model.train_targets.detach().cpu().numpy()      # (N, P)

    lengthscales = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()  # (P, 1, D)
    outputscales = model.covar_module.outputscale.detach().cpu().numpy()              # (P,)
    noise = model.likelihood.noise.detach().cpu().numpy()                              # (P,)

    alpha = []
    X = torch.tensor(train_x)
    for p in range(train_y.shape[1]):
        ls = model.covar_module.base_kernel.lengthscale[p, 0, :]
        os = model.covar_module.outputscale[p]
        y = torch.tensor(train_y[:, p])

        diff = X.unsqueeze(1) - X.unsqueeze(0)
        dist_sq = (diff / ls).pow(2).sum(-1)
        K = os * torch.exp(-0.5 * dist_sq)
        K += torch.eye(len(train_x)) * noise[p]

        alpha_p = torch.linalg.solve(K, y)
        alpha.append(alpha_p.detach().cpu().numpy())

    return train_x, alpha, lengthscales, outputscales

def plot_gp_fit_on_training_data(train_inputs, train_outputs, gp_model, likelihood, task_names=None):
    """
    Plots the GP mean and 95% confidence interval on training data.
    
    Parameters:
    - train_inputs: torch.Tensor of shape (N, D)
    - train_outputs: torch.Tensor of shape (N, T)
    - gp_model: Trained GPyTorch model
    - likelihood: GPyTorch likelihood
    - task_names: Optional list of names for each output dimension (length T)
    """
    gp_model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(gp_model(train_inputs))
    
    mean = pred.mean.cpu()
    lower, upper = pred.confidence_region()

    num_tasks = mean.shape[1]
    x_axis = np.arange(train_inputs.shape[0])
    
    if task_names is None:
        task_names = [f'Task {i}' for i in range(num_tasks)]

    for i in range(num_tasks):
        plt.figure(figsize=(10, 4))
        plt.plot(x_axis, train_outputs[:, i].cpu(), 'k*', label='Training targets')
        plt.plot(x_axis, mean[:, i], 'b', label='GP mean')
        plt.fill_between(
            x_axis,
            lower[:, i],
            upper[:, i],
            alpha=0.3,
            label='95% confidence interval'
        )
        plt.title(f'GP fit on training data - {task_names[i]}')
        plt.xlabel('Training sample index')
        plt.ylabel('Output')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def vertical_points_ref(Ts, N_points, pred_hor):
    #time_ref = np.linspace(0, Ts, N_points+pred_hor)
    time_ref = np.arange(0, Ts*(N_points+pred_hor), Ts)

    # Split indices into three segments, even if not equal length
    idx_splits = np.array_split(np.arange(N_points+pred_hor), 15)

    cart_ref = np.zeros(len(time_ref))
    theta_ref = np.zeros(len(time_ref))
    v_ref = np.zeros(len(time_ref))
    omega_ref = np.zeros(len(time_ref))

    cart_ref[idx_splits[0]] = -1
    cart_ref[idx_splits[1]] = -2
    cart_ref[idx_splits[2]] = -6
    cart_ref[idx_splits[3]] = -4
    cart_ref[idx_splits[4]] = 0
    cart_ref[idx_splits[5]] = 3
    cart_ref[idx_splits[6]] = -3
    cart_ref[idx_splits[7]] = 2
    cart_ref[idx_splits[8]] = -1
    cart_ref[idx_splits[9]] = 5
    cart_ref[idx_splits[10]] = 0
    cart_ref[idx_splits[11]] = -3
    cart_ref[idx_splits[12]] = 0
    cart_ref[idx_splits[13]] = -2
    cart_ref[idx_splits[14]] = 3

    v_ref = np.gradient(cart_ref)

    return time_ref, cart_ref, theta_ref, v_ref, omega_ref

def sinusoidal_ref(Ts, N_points, pred_hor):
    time_ref = np.arange(0, (N_points+pred_hor) * Ts, Ts)

    # Split indices into three segments, even if not equal length
    idx_splits = np.array_split(np.arange(N_points+pred_hor), 10)

    cart_ref = np.zeros(len(time_ref))
    theta_ref = np.zeros(N_points+pred_hor)
    # v_ref = np.zeros(N_points+pred_hor)
    # v_ref = np.gradient(cart_ref)
    omega_ref = np.zeros(N_points+pred_hor)

    theta_ref[idx_splits[0]] = 2*np.sin(3*time_ref[idx_splits[0]])
    theta_ref[idx_splits[1]] = 2*np.sin(time_ref[idx_splits[1]])
    theta_ref[idx_splits[2]] = 2*np.sin(time_ref[idx_splits[2]])
    theta_ref[idx_splits[3]] = 2*np.sin(time_ref[idx_splits[3]])
    theta_ref[idx_splits[4]] = 2*np.sin(time_ref[idx_splits[4]])
    theta_ref[idx_splits[5]] = 2*np.sin(time_ref[idx_splits[5]])
    theta_ref[idx_splits[6]] = 2*np.sin(time_ref[idx_splits[6]])
    theta_ref[idx_splits[7]] = 2*np.sin(time_ref[idx_splits[7]])
    theta_ref[idx_splits[8]] = np.sin(time_ref[idx_splits[8]])
    theta_ref[idx_splits[9]] = 1.1*np.sin(2*time_ref[idx_splits[9]])

    v_ref = np.gradient(cart_ref)
    omega_ref = np.gradient(theta_ref)

    # Middle third: sin(2t)
    # cart_ref[idx_splits[1]] = 2

    # cart_ref[idx_splits[2]] = 6

    return time_ref, cart_ref, theta_ref, v_ref, omega_ref

def mix_ref(Ts, N_points, pred_hor, seed=42):
    np.random.seed(seed)
    time_ref = np.arange(0, (N_points + pred_hor) * Ts, Ts)
    total_len = len(time_ref)

    # Define how many splits to make (more than before)
    N_segments = 15
    idx_splits = np.array_split(np.arange(total_len), N_segments)

    # Preallocate references
    cart_ref = np.zeros(total_len)
    theta_ref = np.zeros(total_len)

    # Define base patterns
    def pattern_cart(i, t):
        freq = 0.5 *(i + 1)
        return 4 * np.sin(0.4* freq * t)

    def pattern_theta(i, t):
        if i % 4 == 0:
            return np.pi * np.ones_like(t)
        elif i % 4 == 1:
            return 2 * np.sin(t)
        elif i % 4 == 2:
            return np.pi * np.sin(t)
        else:
            return np.zeros_like(t)

    # Choose from base patterns randomly and assign to segments
    base_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    repeated_indices = np.random.choice(base_indices, size=N_segments, replace=True)

    for seg_idx, base_i in enumerate(repeated_indices):
        idx = idx_splits[seg_idx]
        t = time_ref[idx]
        cart_ref[idx] = pattern_cart(base_i, t)
        theta_ref[idx] = pattern_theta(base_i, t)

    # Compute derivatives
    v_ref = np.gradient(cart_ref, Ts)
    omega_ref = np.gradient(theta_ref, Ts)

    data_dict = {
        "time_ref": time_ref.copy(),
        "cart_ref": cart_ref.copy(),
        "theta_ref": theta_ref.copy(),
        "v_ref": v_ref.copy(),
        "omega_ref": omega_ref.copy(),
    }

    np.savez("rich_mix_ref_expanded.npz", **data_dict)
    return time_ref, cart_ref, theta_ref, v_ref, omega_ref

def exploratory_ref(Ts, N_points, pred_hor):
    time_ref = np.arange(0, (N_points+pred_hor) * Ts, Ts)
    u_ref = np.zeros(len(time_ref))
    # Split indices into three segments, even if not equal length
    idx_splits = np.array_split(np.arange(N_points+pred_hor), 10)

    u_ref[idx_splits[0]] = -4
    u_ref[idx_splits[1]] = 40*np.sin(time_ref[idx_splits[1]]) 
    u_ref[idx_splits[2]] = abs(30*np.sin(2*time_ref[idx_splits[2]]) )
    u_ref[idx_splits[3]] = 20*np.sin(3*time_ref[idx_splits[3]]) 
    u_ref[idx_splits[4]] = 10*np.sin(4*time_ref[idx_splits[4]]) 
    u_ref[idx_splits[5]] = 4*np.sin(5*time_ref[idx_splits[5]]) 
    u_ref[idx_splits[6]] = 4*np.sin(5*time_ref[idx_splits[6]]) 
    u_ref[idx_splits[7]] = 4*np.sin(4*time_ref[idx_splits[7]]) 
    u_ref[idx_splits[8]] = 4*np.sin(2*time_ref[idx_splits[8]]) 
    u_ref[idx_splits[9]] = 4*np.sin(3*time_ref[idx_splits[9]]) 

    return time_ref, u_ref


def rich_mix_ref(Ts, N_points, pred_hor):
    time_ref = np.arange(0, (N_points+pred_hor) * Ts, Ts)

    # Split indices into three segments, even if not equal length
    idx_splits = np.array_split(np.arange(N_points+pred_hor), 10)

    cart_ref = np.zeros(len(time_ref))
    theta_ref = np.zeros(len(time_ref))
    v_ref = np.zeros(len(time_ref))
    omega_ref = np.zeros(len(time_ref))
    
    cart_ref[idx_splits[1]] = 4*np.sin(time_ref[idx_splits[1]])
    cart_ref[idx_splits[2]] = 4*np.sin(2*time_ref[idx_splits[2]])
    cart_ref[idx_splits[3]] = 4*np.sin(3*time_ref[idx_splits[3]])
    cart_ref[idx_splits[4]] = 4*np.sin(4*time_ref[idx_splits[4]])
    cart_ref[idx_splits[5]] = 4*np.sin(5*time_ref[idx_splits[5]])
    cart_ref[idx_splits[6]] = np.zeros(len(time_ref[idx_splits[0]]))# 4*np.sin(5*time_ref[idx_splits[6]])
    cart_ref[idx_splits[7]] = 4*np.sin(4*time_ref[idx_splits[7]])
    cart_ref[idx_splits[8]] = 4*np.sin(2*time_ref[idx_splits[8]])
    cart_ref[idx_splits[9]] = 4*np.sin(3*time_ref[idx_splits[9]])

    theta_ref[idx_splits[1]] = np.pi * np.ones(len(idx_splits[1]))
    theta_ref[idx_splits[2]] = np.pi * np.ones(len(idx_splits[1]))
    theta_ref[idx_splits[3]] = np.pi*np.sin(time_ref[idx_splits[6]])
    theta_ref[idx_splits[4]] = np.pi*np.sin(time_ref[idx_splits[7]])
    theta_ref[idx_splits[5]] = 0*np.ones(len(idx_splits[1]))
    theta_ref[idx_splits[6]] = np.pi * np.ones(len(idx_splits[1]))
    theta_ref[idx_splits[7]] = 0 * np.ones(len(idx_splits[1]))
    theta_ref[idx_splits[8]] = 2*np.sin(time_ref[idx_splits[8]])
    theta_ref[idx_splits[9]] = 2*np.sin(time_ref[idx_splits[9]])

    v_ref = np.gradient(cart_ref)
    omega_ref = np.gradient(theta_ref)
    # cart_ref[idx_splits[10]] = np.sin(15*time_ref[idx_splits[10]])   
    data_dict = {
        "time_ref" : time_ref.copy(),
        "cart_ref" : cart_ref.copy(),
        "theta_ref" : theta_ref.copy(),
        "v_ref" : v_ref.copy(), 
        "omega_ref" : omega_ref.copy(),
    } 
    print
    np.savez("rich_mix_ref.npz", **data_dict)
    return time_ref, cart_ref, theta_ref, v_ref, omega_ref



def plot_references(time_ref, cart_ref, theta_ref, v_ref, omega_ref):
    """
    Plot reference signals over time.

    Parameters:
        time_ref   (array): Time vector
        cart_ref   (array): Cart position reference
        theta_ref  (array): Pendulum angle reference
        v_ref      (array): Cart velocity reference
        omega_ref  (array): Pendulum angular velocity reference
        slack_ref  (array): Slack variable reference
    """
    plt.figure(figsize=(12, 10))

    plt.subplot(5, 1, 1)
    plt.plot(time_ref, cart_ref, label='Cart Position (ref)')
    plt.ylabel('Cart')
    plt.legend()
    plt.grid(True)

    plt.subplot(5, 1, 2)
    plt.plot(time_ref, theta_ref, label='Pendulum Angle (ref)', color='orange')
    plt.ylabel('Theta')
    plt.legend()
    plt.grid(True)

    plt.subplot(5, 1, 3)
    plt.plot(time_ref, v_ref, label='Cart Velocity (ref)', color='green')
    plt.ylabel('Velocity')
    plt.legend()
    plt.grid(True)

    plt.subplot(5, 1, 4)
    plt.plot(time_ref, omega_ref, label='Angular Velocity (ref)', color='purple')
    plt.ylabel('Omega')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def create_gp_input_from_npz(filenames):
    # filenames = ["gp_data01.npz", "gp_data02.npz", "gp_data03.npz"]

    # Initialize empty lists to hold data from all files
    X_gp_list = []
    Y_gp_v_list = []
    Y_gp_w_list = []

    # Load and append data from each file
    for file in filenames:
        data = np.load(file)

        X = data['X_gp']
        Y_v = data['Y_gp_v']
        Y_w = data['Y_gp_w']
        X_gp_list.append(X[:-1, :])  # Exclude the last row
        Y_gp_v_list.append(Y_v)
        Y_gp_w_list.append(Y_w)
    # Concatenate all data along the first axis (rows)
    X_gp_merged = np.concatenate(X_gp_list, axis=0)
    Y_gp_v_merged = np.concatenate(Y_gp_v_list, axis=0)
    Y_gp_w_merged = np.concatenate(Y_gp_w_list, axis=0)

    return X_gp_merged, Y_gp_v_merged, Y_gp_w_merged

def visualize_inverted_pendulum(X_sim, U_sim, time_vec, REF=None):
    """
    Real-time interactive animation of an inverted pendulum on a cart.

    Parameters:
    - X_sim: (T, 4) numpy array [x, theta, x_dot, theta_dot]
    - U_sim: (T, 1) or (T,) numpy array, control inputs (not visualized)
    - time_vec: (T,) numpy array, time stamps
    - REF: (6,) or (T, 6) numpy array, optional reference trajectory
    """
    # Parameters
    L = 0.5               # Pendulum length (m)
    cart_width = 0.3
    cart_height = 0.2
    wheel_radius = 0.05

    x = X_sim[:, 0]
    theta = X_sim[:, 1]
    time = time_vec
    T = len(time)

    # Reference handling    
    if REF is None:
        REF = np.zeros((T, 6))
        

    elif REF.ndim == 1:
        # REF è un vettore 1D: trattato come solo riferimento posizione carrello
        ref_len = REF.shape[0]
        if ref_len > T:
            REF = REF[:T]
        elif ref_len < T:
            REF = np.pad(REF, (0, T - ref_len), mode='constant')
    print("REF shape is ", REF.shape)

    # Figure setup
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim([-7, 7])
    ax.set_ylim([-1, 1])
    ax.set_title('Inverted Pendulum on a Cart')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True)

    # Patches and lines
    cart_patch = patches.Rectangle((x[0] - cart_width/2, -cart_height/2),
                                   cart_width, cart_height, color=[0.2, 0.6, 1])
    wheel1 = patches.Ellipse((x[0] - cart_width/2 + 0.1 + wheel_radius, -cart_height/2 - wheel_radius),
                              2*wheel_radius, wheel_radius, color='k')
    wheel2 = patches.Ellipse((x[0] + cart_width/2 - 0.1 - wheel_radius, -cart_height/2 - wheel_radius),
                              2*wheel_radius, wheel_radius, color='k')
    pendulum_line, = ax.plot([], [], 'r', linewidth=3)
    cart_ref_marker, = ax.plot([], [], 'gx', markersize=10, linewidth=2)
    pendulum_ref_marker, = ax.plot([], [], 'bx', markersize=10, linewidth=2)
    timestamp = ax.text(-3.8, 0.65, '', fontsize=12, fontweight='bold', color='k')

    ax.add_patch(cart_patch)
    ax.add_patch(wheel1)
    ax.add_patch(wheel2)

    # Animation control variables
    paused = [False]
    speed_factor = [1.0]

    def update_frame(k):
        cart_patch.set_xy((x[k] - cart_width / 2, -cart_height / 2))
        wheel1.center = (x[k] - cart_width/2 + 0.1 + wheel_radius, -cart_height/2 - wheel_radius)
        wheel2.center = (x[k] + cart_width/2 - 0.1 - wheel_radius, -cart_height/2 - wheel_radius)

        px = x[k] - L * np.sin(theta[k])
        py = +L * np.cos(theta[k])
        pendulum_line.set_data([x[k], px], [0, py])

        cart_ref_marker.set_data(REF[k], 0)
        #pendulum_ref_marker.set_data(REF[k, 0] - L * np.sin(REF[k, 1]),
        #                              L * np.cos(REF[k, 1]))

        timestamp.set_text(f'Time: {time[k]:.2f} s')
        return (cart_patch, wheel1, wheel2, pendulum_line,
                cart_ref_marker, pendulum_ref_marker, timestamp)

    # Frame index (manual control for pause)
    frame_idx = [0]

    def update(_):
        if not paused[0]:
            update_frame(frame_idx[0])
            frame_idx[0] = (frame_idx[0] + 1) % T
        return []

    # Key press handler
    def on_key(event):
        if event.key == ' ':
            paused[0] = not paused[0]
        elif event.key == '+':
            speed_factor[0] = min(speed_factor[0] * 1.5, 10)
        elif event.key == '-':
            speed_factor[0] = max(speed_factor[0] / 1.5, 0.1)
        print(f"Paused: {paused[0]}, Speed factor: {speed_factor[0]:.2f}")

    fig.canvas.mpl_connect('key_press_event', on_key)

    # Create timer-based animation
    interval_ms = int(1000 * np.mean(np.diff(time)) / speed_factor[0])
    ani = animation.FuncAnimation(fig, update, interval=interval_ms, blit=False)

    plt.show()


# 1. Build the augmented state for the controller
def build_augmented_state(state_hist, input_hist):
    # state_hist[-1] is current, [-2] is k-1, [-3] is k-2
    x_k     = state_hist[-1]
    x_k1    = state_hist[-2]
    x_k2    = state_hist[-3]
    u_k1    = input_hist[-2]
    u_k2    = input_hist[-3]
    x_aug = np.concatenate([x_k, x_k1[0:2], u_k1, x_k2[0:2], u_k2])  # Only x, theta and inputs from past
    return x_aug

def load_gp_data_from_csv(sim_file, n_outputs):
    """
    Loads data from the specified CSV files and returns the content as NumPy arrays.

    Parameters:
    - sim_file (str): Path to the CSV file with X_sim and GP targets.
    - gp_file (str): Path to the CSV file with X_gp and GP targets.

    Returns:
    - X_sim (ndarray): State trajectory from simulation (without last step).
    - X_gp (ndarray): GP input states and features.
    - Y_gp (ndarray): GP targets [Δp, Δθ, Δv, Δω].
    """
    # Load data from CSV
    data = np.loadtxt(sim_file, delimiter=',', skiprows=1)

    # Assume targets are always in the last 4 columns
    X = data[:, :-n_outputs]
    Y = data[:, -n_outputs:]  # Or gp_data[:, -4:] — they are the same     

    # Load CSV and convert to tensor
    X = torch.tensor(X, dtype=torch.float)
    Y = torch.tensor(Y, dtype=torch.float)

    return X, Y

def filter_dataset(X, Y, min_distance=1e-1):

    print("X before filtering is of shape ", X.shape)
    print("Y before filtering is of shape ", Y.shape)

    # Use NearestNeighbors to find neighbors within the distance
    nbrs = NearestNeighbors(radius=min_distance, algorithm='ball_tree')
    print("LEN NBRS ", (nbrs))
    nbrs.fit(X)
    radii_neighbors = nbrs.radius_neighbors(X, return_distance=False)

    # Greedy filtering: keep first point, remove others in its neighborhood
    keep_mask = torch.ones(len(X), dtype=torch.bool)

    for i in range(len(X)):
        if keep_mask[i]:
            neighbors = radii_neighbors[i]
            # Mark all neighbors (except itself) for removal
            for j in neighbors:
                if j != i:
                    keep_mask[j] = False

    X_filtered = X[keep_mask]
    Y_filtered = Y[keep_mask]

    X_sim_and_targets = np.hstack([X_filtered, Y_filtered])
    np.savetxt("X_sim_filtered.csv", X_sim_and_targets, delimiter=',', header='x1,x2,x3,x4,u,Y_p,Y_theta,Y_v,Y_w', comments='')

    print("X after filtering is of shape ", X_filtered.shape)
    print("Y after filtering is of shape ", Y_filtered.shape)

    return X_filtered, Y_filtered

def check_cov_matrix(train_inputs):
    print("Computing rank")
    kernel = gpytorch.kernels.RBFKernel(batch_shape=torch.Size([4]))
    K = kernel(train_inputs, train_inputs).evaluate()
    rank = torch.linalg.matrix_rank(K)
    print("Shape of K matrix is ", K.shape)
    # Check se è simmetrica (deve esserlo)
    for i in range(K.shape[0]):
        print("Matrice simmetrica? ", torch.allclose(K[i,:,:], K[i,:,:].T, atol=1e-5))
    
    # Prova la decomposizione di Cholesky (fallisce se non è PD)
    print("Trying Cholesky decomposition...")
    try:
        L = torch.linalg.cholesky(K)
        print("Matrice positiva definita ✅")
    except RuntimeError:
        print("Matrice NON positiva definita ❌")
    
    print(rank, "/", K.size(-1))  # full rank?
    
    _, unique_indices = torch.unique(train_inputs, dim=0, return_inverse=True)
    print(f"dimensione train input {train_inputs.size(0)}\n unique indices {len(unique_indices)}")
    if len(unique_indices) < train_inputs.size(0):
        print("Ci sono dati duplicati ⚠️")

def augment_csv_by_column_variation(input_csv_path, output_csv_path, column_index, values):
    """
    Duplica ogni riga del CSV (escluso l'header) variando un solo elemento (colonna).
    
    Args:
        input_csv_path (str): Percorso del file CSV di input.
        output_csv_path (str): Percorso dove salvare il nuovo file CSV.
        column_index (int): Indice della colonna da modificare (0-based).
        values (list or array): Valori da assegnare alla colonna modificata.
    """
    # Carica il file CSV con header
    df = pd.read_csv(input_csv_path)

    # Salva separatamente l'intestazione
    header = df.columns
    data = df.copy()

    # Lista per le righe duplicate
    augmented_rows = []

    # Duplica ogni riga e modifica la colonna specificata
    for _, row in data.iterrows():
        for val in values:
            new_row = row.copy()
            new_row.iloc[column_index] = val
            augmented_rows.append(new_row)

    # Crea il DataFrame finale
    df_augmented = pd.DataFrame(augmented_rows, columns=header)

    # Salva il file CSV con header
    df_augmented.to_csv(output_csv_path, header=True, index=False)

def get_SOD(self, X, Y, threshold, flg_permutation=False):
    """
    Returns the SOD points with an online procedure
    SOD: most importants subset of data
    """
    print('\nSelection of the inducing inputs...')
    # get number of samples
    num_samples = X.shape[0]
    # init the set of inducing inputs with the first sample
    SOD = X[0:1,:]
    inducing_inputs_indices = [0]
    # get a permuation of the inputs
    # perm_indices = torch.arange(1,num_samples)
    perm_indices = range(1,num_samples)
    if flg_permutation:
        perm_indices = perm_indices[torch.randperm(num_samples-1)]
    # iterate all the samples
    for sample_index in perm_indices:
        # get the estimate 
        _, var, _ = self.get_estimate(X[inducing_inputs_indices,:], Y[inducing_inputs_indices,:], X[sample_index:sample_index+1,:])
        if torch.sqrt(var)>=threshold:
            SOD = torch.cat([SOD,X[sample_index:sample_index+1,:]],0)
            inducing_inputs_indices.append(sample_index)
        # else:
        #     print('torch.sqrt(var) = ',torch.sqrt(var))
        #     print('threshold = ',threshold)
    print('Shape inducing inputs selected:', SOD.shape)
    return inducing_inputs_indices

def delete_json_files():
    # Get a list of all .json files in the current directory
    json_files = glob.glob("*.json")

    # Loop through the list and delete each file
    for file in json_files:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")

def open_npz_file(file_path):

    if not os.path.exists(file_path):
        raise FileNotFoundError(f'File {file_path} does not exist.')
    
    try:
        data = np.load(file_path, allow_picke=True)
        print(f'loaded {file_path} with keys: {list(data.keys())}')

        return data
    except Exception as e:
        raise RuntimeError(f'Error loading {file_path}: {e}')
    
    