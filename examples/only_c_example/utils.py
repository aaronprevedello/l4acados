import numpy as np
from numpy import linalg as npla
from dataclasses import dataclass
import matplotlib.pyplot as plt
from pendulum_model import export_pendulum_ode_model_with_discrete_rk4
import torch

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
