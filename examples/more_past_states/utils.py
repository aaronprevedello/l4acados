import numpy as np
from numpy import linalg as npla
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

import torch, gpytorch

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

def export_gp_to_c(train_x, alpha, lengthscales, outputscales, file_path="gp_dynamics.c"):
    with open(file_path, "w") as f:
        N = train_x.shape[0]
        D = train_x.shape[1]
        P = alpha.shape[0]

        f.write('#include "acados/utils/types.h"\n')
        f.write('#include <math.h>\n\n')
        f.write('int gp_dynamics(const real_t** in, real_t** out, int* iw, real_t* w, void *mem) {\n')
        f.write('    const real_t *zu = in[0];\n')
        f.write('    real_t *x_next = out[0];\n')

        f.write(f'    // GP model: N={N}, D={D}, P={P}\n')

        # Write training data
        f.write(f'    const real_t train_x[{N}][{D}] = {{\n')
        for x in train_x:
            f.write("        {" + ", ".join(f"{v:.8f}" for v in x) + "},\n")
        f.write("    };\n")

        # Alpha
        f.write(f'    const real_t alpha[{P}][{N}] = {{\n')
        for p in range(P):
            f.write("        {" + ", ".join(f"{v:.8f}" for v in alpha[p]) + "},\n")
        f.write("    };\n")

        # Lengthscales
        f.write(f'    const real_t lengthscale[{P}][{D}] = {{\n')
        for p in range(P):
            f.write("        {" + ", ".join(f"{v:.8f}" for v in lengthscales[p, 0]) + "},\n")
        f.write("    };\n")

        # Outputscale
        f.write(f'    const real_t outputscale[{P}] = {{{", ".join(f"{v:.8f}" for v in outputscales)}}};\n\n')

        # Start prediction loop
        f.write('    for (int p = 0; p < {0}; p++) {{\n'.format(P))
        f.write('        real_t mean = 0.0;\n')
        f.write('        for (int i = 0; i < {0}; i++) {{\n'.format(N))
        f.write('            real_t dist = 0.0;\n')
        f.write('            for (int d = 0; d < {0}; d++) {{\n'.format(D))
        f.write('                real_t diff = (zu[d] - train_x[i][d]) / lengthscale[p][d];\n')
        f.write('                dist += diff * diff;\n')
        f.write('            }\n')
        f.write('            mean += alpha[p][i] * outputscale[p] * exp(-0.5 * dist);\n')
        f.write('        }\n')
        f.write('        x_next[p] = mean;\n')
        f.write('    }\n')

        f.write('    return 0;\n}\n')

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
    idx_splits = np.array_split(np.arange(N_points+pred_hor), 6)

    cart_ref = np.zeros(len(time_ref))
    theta_ref = np.zeros(len(time_ref))
    v_ref = np.zeros(len(time_ref))
    omega_ref = np.zeros(len(time_ref))

    # Middle third: sin(2t)
    cart_ref[idx_splits[1]] = -2

    cart_ref[idx_splits[2]] = -6

    cart_ref[idx_splits[3]] = -2

    cart_ref[idx_splits[4]] = 0
    cart_ref[idx_splits[5]] = -1

    return time_ref, cart_ref, theta_ref, v_ref, omega_ref

def sinusoidal_ref(Ts, N_points, pred_hor):
    time_ref = np.arange(0, (N_points+pred_hor) * Ts, Ts)

    # Split indices into three segments, even if not equal length
    idx_splits = np.array_split(np.arange(N_points+pred_hor), 3)

    cart_ref = np.sin(time_ref)
    theta_ref = np.zeros(N_points+pred_hor)
    # v_ref = np.zeros(N_points+pred_hor)
    # v_ref = np.gradient(cart_ref)
    omega_ref = np.zeros(N_points+pred_hor)

    cart_ref[idx_splits[1]] = np.sin(3*time_ref[idx_splits[1]])
    cart_ref[idx_splits[2]] = np.sin(7*time_ref[idx_splits[2]])

    v_ref = np.gradient(cart_ref)

    # Middle third: sin(2t)
    # cart_ref[idx_splits[1]] = 2

    # cart_ref[idx_splits[2]] = 6

    return time_ref, cart_ref, theta_ref, v_ref, omega_ref

def mix_ref(Ts, N_points, pred_hor):
    time_ref = np.arange(0, (N_points+pred_hor) * Ts, Ts)

    # Split indices into three segments, even if not equal length
    idx_splits = np.array_split(np.arange(N_points+pred_hor), 10)

    cart_ref = np.zeros(len(time_ref))
    theta_ref = np.zeros(len(time_ref))
    v_ref = np.zeros(len(time_ref))
    omega_ref = np.zeros(len(time_ref))
    # Middle third: sin(2t)
    cart_ref[idx_splits[1]] = -2
    cart_ref[idx_splits[2]] = -6
    cart_ref[idx_splits[3]] = -2
    cart_ref[idx_splits[4]] = 0
    cart_ref[idx_splits[5]] = -1
    cart_ref[idx_splits[6]] = 0

    cart_ref[idx_splits[7]] = 3*np.sin(3*time_ref[idx_splits[7]])
    cart_ref[idx_splits[8]] = 3*np.sin(5*time_ref[idx_splits[8]])
    cart_ref[idx_splits[9]] = 3*np.sin(time_ref[idx_splits[9]])

    v_ref = np.gradient(cart_ref)
    # cart_ref[idx_splits[10]] = np.sin(15*time_ref[idx_splits[10]])   
    data_dict = {
        "time_ref" : time_ref.copy(),
        "cart_ref" : cart_ref.copy(),
        "theta_ref" : theta_ref.copy(),
        "v_ref" : v_ref.copy(), 
        "omega_ref" : omega_ref.copy(),
    } 
    np.savez("mix_ref.npz", **data_dict)
    return time_ref, cart_ref, theta_ref, v_ref, omega_ref

import numpy as np

def rich_mix_ref(Ts, N_points, pred_hor):
    time_ref = np.arange(0, (N_points + pred_hor) * Ts, Ts)
    n_segments = 14
    idx_splits = np.array_split(np.arange(N_points + pred_hor), n_segments)

    cart_ref = np.zeros(len(time_ref))
    theta_ref = np.zeros(len(time_ref))
    omega_ref = np.zeros(len(time_ref))

    # Cart reference: combina step, rampe, sinusoidi
    cart_ref[idx_splits[0]]  = -3
    cart_ref[idx_splits[1]]  = -5
    cart_ref[idx_splits[2]]  = np.linspace(-5, 5, len(idx_splits[2]))  # rampa
    cart_ref[idx_splits[3]]  = 2
    cart_ref[idx_splits[4]]  = 3 * np.sin(2 * time_ref[idx_splits[4]])
    cart_ref[idx_splits[5]]  = -3 * np.sin(3 * time_ref[idx_splits[5]])
    cart_ref[idx_splits[6]]  = 4 * np.sin(0.5 * time_ref[idx_splits[6]])
    cart_ref[idx_splits[7]]  = np.linspace(2, -2, len(idx_splits[7]))
    cart_ref[idx_splits[8]]  = 0
    cart_ref[idx_splits[9]]  = 1.5 * np.sin(4 * time_ref[idx_splits[9]])
    cart_ref[idx_splits[10]] = 2 * np.sign(np.sin(2 * time_ref[idx_splits[10]]))  # onda quadra
    cart_ref[idx_splits[11]] = np.random.uniform(-2, 2, len(idx_splits[11]))      # random walk morbido
    cart_ref[idx_splits[12]] = 1.5 * np.cos(time_ref[idx_splits[12]])
    cart_ref[idx_splits[13]] = 0.5 * np.sin(6 * time_ref[idx_splits[13]])

    # Theta reference: comportamenti diversi per il pendolo
    #theta_ref[idx_splits[0]]  = 0.2
    #theta_ref[idx_splits[1]]  = -0.3
    #theta_ref[idx_splits[2]]  = np.linspace(0, np.pi/4, len(idx_splits[2]))      # inclinazione crescente
    #theta_ref[idx_splits[3]]  = 0
    #theta_ref[idx_splits[4]]  = 0.5 * np.sin(3 * time_ref[idx_splits[4]])
    #theta_ref[idx_splits[5]]  = 0.3 * np.cos(2 * time_ref[idx_splits[5]])
    #theta_ref[idx_splits[6]]  = -0.4 * np.sin(1.5 * time_ref[idx_splits[6]])
    #theta_ref[idx_splits[7]]  = 0.3 * np.sin(5 * time_ref[idx_splits[7]])
    #theta_ref[idx_splits[8]]  = np.linspace(np.pi/6, -np.pi/6, len(idx_splits[8]))
    #theta_ref[idx_splits[9]]  = 0.2 * np.sign(np.sin(2 * time_ref[idx_splits[9]])) # onde quadre
    #theta_ref[idx_splits[10]] = np.random.uniform(-0.3, 0.3, len(idx_splits[10]))
    #theta_ref[idx_splits[11]] = 0
    #theta_ref[idx_splits[12]] = 0.4 * np.sin(3 * time_ref[idx_splits[12]])
    #theta_ref[idx_splits[13]] = 0

    # Derivate numeriche per ottenere velocità
    v_ref = np.gradient(cart_ref, Ts)
    #omega_ref = np.gradient(theta_ref, Ts)

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

#def visualize_inverted_pendulum(X_sim, U_sim, time_vec, REF=None):
#    """
#    Visualize an inverted pendulum on a cart in real-time.
#
#    Parameters:
#    - X_sim: np.ndarray, shape (T, 4), state over time [x, theta, x_dot, theta_dot]
#    - U_sim: np.ndarray, shape (T, 1) or (T,), control input (not used in plot)
#    - time_vec: np.ndarray, shape (T,), time values
#    - REF: np.ndarray, shape (6,) or (T, 6), optional reference signal
#    """
#    # Constants
#    L = 0.5               # Pendulum length (m)
#    cart_width = 0.3      # Cart width (m)
#    cart_height = 0.2     # Cart height (m)
#    wheel_radius = 0.05   # Wheel radius (m)
#    pendulum_width = 0.8  # Not directly used
#
#    x = X_sim[:, 0]
#    theta = X_sim[:, 1]
#    time = time_vec
#    T = len(time)
#
#    # Handle reference input
#    if REF is None:
#        REF = np.zeros((T, 6))
#    elif REF.shape == (6,):
#        REF = np.tile(REF, (T, 1))
#    elif REF.shape[0] != T:
#        raise ValueError("REF must have shape (6,) or (T, 6)")
#
#    # Set up the figure
#    fig, ax = plt.subplots()
#    ax.set_aspect('equal')
#    ax.grid(True)
#    ax.set_xlim([-6, 6])
#    ax.set_ylim([-0.75, 0.75])
#    ax.set_title('Inverted Pendulum on a Cart')
#    ax.set_xlabel('X (m)')
#    ax.set_ylabel('Y (m)')
#
#    # Initialize drawing elements
#    cart_patch = patches.Rectangle((x[0] - cart_width/2, -cart_height/2),
#                                   cart_width, cart_height, color=[0.2, 0.6, 1])
#    wheel1_patch = patches.Ellipse((x[0] - cart_width/2 + 0.1 + wheel_radius, -cart_height/2 - wheel_radius),
#                                   2*wheel_radius, wheel_radius, color='k')
#    wheel2_patch = patches.Ellipse((x[0] + cart_width/2 - 0.1 - wheel_radius, -cart_height/2 - wheel_radius),
#                                   2*wheel_radius, wheel_radius, color='k')
#    pendulum_line, = ax.plot([], [], 'r', linewidth=3)
#    cart_ref_marker, = ax.plot([], [], 'gx', markersize=10, linewidth=2)
#    pendulum_ref_marker, = ax.plot([], [], 'bx', markersize=10, linewidth=2)
#    timestamp_text = ax.text(-3.8, 0.65, '', fontsize=12, fontweight='bold', color='k')
#
#    # Add patches to the axes
#    ax.add_patch(cart_patch)
#    ax.add_patch(wheel1_patch)
#    ax.add_patch(wheel2_patch)
#
#    # Frame update function
#    def update_frame(k):
#        cart_patch.set_xy((x[k] - cart_width / 2, -cart_height / 2))
#        wheel1_patch.center = (x[k] - cart_width/2 + 0.1 + wheel_radius, -cart_height/2 - wheel_radius)
#        wheel2_patch.center = (x[k] + cart_width/2 - 0.1 - wheel_radius, -cart_height/2 - wheel_radius)
#
#        px = x[k] - L * np.sin(theta[k])
#        py = L * np.cos(theta[k])
#        pendulum_line.set_data([x[k], px], [0, py])
#
#        cart_ref_marker.set_data(REF[k, 0], 0)
#        pendulum_ref_marker.set_data(REF[k, 0] - L * np.sin(REF[k, 1]),
#                                    L * np.cos(REF[k, 1]))
#
#        timestamp_text.set_text(f'Time: {time[k]:.2f} s')
#
#        return (cart_patch, wheel1_patch, wheel2_patch,
#                pendulum_line, cart_ref_marker, pendulum_ref_marker, timestamp_text)
#
#    # Create and display animation
#    ani = animation.FuncAnimation(fig, update_frame, frames=T, interval=1000 * np.mean(np.diff(time)),
#                                  blit=True, repeat=False)
#
#    plt.show()


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
    elif REF.shape == (6,):
        REF = np.tile(REF, (T, 1))
    elif REF.shape[0] != T:
        raise ValueError("REF must have shape (6,) or (T, 6)")

    # Figure setup
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim([-4, 4])
    ax.set_ylim([-0.75, 0.75])
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

        cart_ref_marker.set_data(REF[k, 0], 0)
        pendulum_ref_marker.set_data(REF[k, 0] - L * np.sin(REF[k, 1]),
                                      L * np.cos(REF[k, 1]))

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
    x_k3    = state_hist[-4]
    x_k4    = state_hist[-5]
    u_k1    = input_hist[-2]
    u_k2    = input_hist[-3]
    u_k3    = input_hist[-4]
    u_k4    = input_hist[-5]
    x_aug = np.concatenate([x_k, x_k1[0:2], u_k1, x_k2[0:2], u_k2, x_k3[0:2], u_k3, x_k4[0:2], u_k4])  # Only x, theta and inputs from past
    return x_aug