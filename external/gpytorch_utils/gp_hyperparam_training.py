import sys, os
import numpy as np
import gpytorch
import torch
import random
import matplotlib.pyplot as plt



def generate_train_inputs_zoro(
    zoro_solver, x0_nom, N_sim_per_x0, N_x0, random_seed=None, x0_rand_scale=0.1
):
    if random_seed is not None:
        np.random.seed(random_seed)

    nx = zoro_solver.nx
    nu = zoro_solver.nu
    N = zoro_solver.N

    x0_arr = np.zeros((N_x0, nx))
    X_inp = np.zeros((N_x0 * N_sim_per_x0 * N, nx + nu))

    i_fac = N_sim_per_x0 * N
    j_fac = N
    for i in range(N_x0):
        x0_arr[i, :] = x0_nom + x0_rand_scale * (2 * np.random.rand(nx) - 1)

        zoro_solver.ocp_solver.set(0, "lbx", x0_arr[i, :])
        zoro_solver.ocp_solver.set(0, "ubx", x0_arr[i, :])
        zoro_solver.solve()

        X, U, P = zoro_solver.get_solution()

        # store training points
        for j in range(N_sim_per_x0):
            for k in range(N):
                ijk = i * i_fac + j * j_fac + k
                X_inp[ijk, :] = np.hstack((X[k, :], U[k, :]))

    return X_inp, x0_arr


def generate_train_inputs_acados(
    ocp_solver, x0_nom, N_sim_per_x0, N_x0, random_seed=None, x0_rand_scale=0.1
):
    if random_seed is not None:
        np.random.seed(random_seed)

    nx = ocp_solver.acados_ocp.dims.nx
    nu = ocp_solver.acados_ocp.dims.nu
    N = ocp_solver.acados_ocp.dims.N

    x0_arr = np.zeros((N_x0, nx))
    X_inp = np.zeros((N_x0 * N_sim_per_x0 * N, nx + nu))

    i_fac = N_sim_per_x0 * N
    j_fac = N
    for i in range(N_x0):
        x0_arr[i, :] = x0_nom + x0_rand_scale * (2 * np.random.rand(nx) - 1)

        ocp_solver.set(0, "lbx", x0_arr[i, :])
        ocp_solver.set(0, "ubx", x0_arr[i, :])
        ocp_solver.solve()

        # store training points
        for j in range(N_sim_per_x0):
            for k in range(N):
                ijk = i * i_fac + j * j_fac + k
                X_inp[ijk, :] = np.hstack(
                    (ocp_solver.get(k, "x"), ocp_solver.get(k, "u"))
                )

    return X_inp, x0_arr


def generate_train_data_acados(
    acados_ocp_solver,
    integrator_nom,
    integrator_sim,
    x0_nom,
    Sigma_W,
    N_sim_per_x0,
    N_sim,
    B=None,
    N_x0=1,
    random_seed=None,
    x0_rand_scale=0.1,
):
    if random_seed is not None:
        np.random.seed(random_seed)

    if B is None:
        B = np.eye(nw)
    B_inv = np.linalg.pinv(B)

    nx = acados_ocp_solver.acados_ocp.model.x.size()[0]
    nu = acados_ocp_solver.acados_ocp.model.u.size()[0]
    nw = Sigma_W.shape[0]

    B_inv = np.linalg.pinv(B)
    x0_arr = np.zeros((N_x0, nx))
    X_inp = np.zeros((N_x0 * N_sim_per_x0 * N_sim, nx + nu))
    Y_out = np.zeros((N_x0 * N_sim_per_x0 * N_sim, nw))

    i_fac = N_sim_per_x0 * N_sim
    j_fac = N_sim
    for i in range(N_x0):
        ijk = i * i_fac
        x0_arr[i, :] = x0_nom + x0_rand_scale * (2 * np.random.rand(nx) - 1)
        xcurrent = x0_arr[i, :]
        for j in range(N_sim_per_x0):
            for k in range(N_sim):
                acados_ocp_solver.set(0, "lbx", xcurrent)
                acados_ocp_solver.set(0, "ubx", xcurrent)
                acados_ocp_solver.solve()

                u = acados_ocp_solver.get(0, "u")

                # integrate nominal model
                integrator_nom.set("x", xcurrent)
                integrator_nom.set("u", u)
                integrator_nom.solve()
                xnom = integrator_nom.get("x")

                # integrate real model
                integrator_sim.set("x", xcurrent)
                integrator_sim.set("u", u)
                integrator_sim.solve()
                xcurrent = integrator_sim.get("x")

                # difference
                w = np.random.multivariate_normal(np.zeros((nw,)), Sigma_W)

                # store training points
                ijk = i * i_fac + j * j_fac + k
                X_inp[ijk, :] = np.hstack((xcurrent, u))

                Y_out[ijk, :] = B_inv @ (xcurrent - xnom) + w

    return X_inp, Y_out


def generate_train_outputs_at_inputs(
    X_inp, integrator_nom, integrator_sim, Sigma_W, B=None
):

    nx = integrator_nom.acados_sim.model.x.size()[0]
    nu = integrator_nom.acados_sim.model.u.size()[0]
    nw = Sigma_W.shape[0]

    if B is None:
        B = np.eye(nw)
    B_inv = np.linalg.pinv(B)

    n_train = X_inp.shape[0]
    Y_out = np.zeros((n_train, nw))
    for i in range(n_train):
        integrator_sim.set("x", X_inp[i, 0:nx])
        integrator_sim.set("u", X_inp[i, nx : nx + nu])
        integrator_sim.solve()

        integrator_nom.set("x", X_inp[i, 0:nx])
        integrator_nom.set("u", X_inp[i, nx : nx + nu])
        integrator_nom.solve()

        w = np.random.multivariate_normal(np.zeros((nw,)), Sigma_W)

        Y_out[i, :] = B_inv @ (integrator_sim.get("x") - integrator_nom.get("x")) + w
    return Y_out


def train_gp_model(
    gp_model, torch_seed=None, training_iterations=200, learning_rate=0.1, val_x=None, val_y=None, compute_val_loss=False
):
    if torch_seed is not None:
        torch.manual_seed(torch_seed)

    train_losses = []
    if compute_val_loss:
        val_losses = []

    likelihood = gp_model.likelihood
    train_x = gp_model.train_inputs[0]
    train_y = gp_model.train_targets

    # Find optimal model hyperparameters
    gp_model.train() # only sets the model in training mode
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, gp_model.parameters()), lr=learning_rate
    )  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

    prev_loss = np.inf
    num_loss_below_threshold = 0

    for i in range(training_iterations):
        gp_model.train()
        likelihood.train()
        optimizer.zero_grad()
        output = (gp_model(train_x))
        loss = -mll(output, train_y.reshape((train_y.numel(),)))

        loss = loss.mean() # added by Aaron because output of inducingpointsgp is a vector
        #import pdb;pdb.set_trace() # aggiunto da me !!!!!!
        num_loss_below_threshold = (
            num_loss_below_threshold + 1 if abs(loss - prev_loss) < 5e-4 else 0
        )

        if num_loss_below_threshold > 5:
            print(f"stopping GP optimization early after {i} iterations.")
            break

        prev_loss = loss

        loss.backward()
        
        optimizer.step()
        if (i + 1) % 20 == 0:
                    print("Iter %d/%d - Loss: %.3f" % (i + 1, training_iterations, loss.item()))
                    train_losses.append(loss.item())
                    if compute_val_loss and val_x is not None and val_y is not None:
                                    val_loss = compute_gp_val_loss(gp_model, likelihood, val_x, val_y)
                                    print(f" - Val Loss: {val_loss:.3f}")
                                    val_losses.append(val_loss.item())
                    else:
                                    print()

    plt.figure()
    plt.plot(np.arange(0, training_iterations, 20), train_losses, label="train loss")
    if compute_val_loss:
        plt.plot(np.arange(0, training_iterations, 20), val_losses, label="val loss")
    plt.show()

    gp_model.eval()
    likelihood.eval()
    return gp_model, likelihood

def compute_gp_val_loss(gp_model, likelihood, val_x, val_y):
    gp_model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = likelihood(gp_model(val_x))
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)
        val_loss = -mll(preds, val_y.view(-1))
        val_loss = val_loss.mean()
        return val_loss

def set_gp_param_value(gp_model, name: str, value: torch.Tensor):
    constraint = gp_model.constraint_for_parameter_name(name)
    parameter = gp_model.get_parameter(name)
    state_dict = gp_model.state_dict()

    # transform value
    if constraint is not None:
        value_transform = constraint.inverse_transform(value)
    else:
        value_transform = value

    state_dict[name] = value_transform
    gp_model.load_state_dict(state_dict)

    return parameter


def get_gp_param_value(gp_model, name: str):
    constraint = gp_model.constraint_for_parameter_name(name)
    parameter = gp_model.get_parameter(name)
    state_dict = gp_model.state_dict()
    value_transform = state_dict[name]

    # transform value
    if constraint is not None:
        value = constraint.transform(value_transform)
    else:
        value = value_transform
    return value


def get_gp_param_names(gp_model):
    names = []
    for name, parameter in gp_model.named_parameters():
        names += [name]
    return names


def get_gp_param_names_values(gp_model):
    names = get_gp_param_names(gp_model)
    values = []
    for name in names:
        values += [get_gp_param_value(gp_model, name)]
    return zip(names, values)


def get_prior_covariance(gp_model):

    dim = gp_model.train_inputs[0].shape[1]

    # cuda check
    if gp_model.train_inputs[0].device.type == "cuda":
        to_numpy = lambda T: T.cpu().numpy()
        y_test = torch.Tensor(np.ones((1, dim))).cuda()
    else:
        to_numpy = lambda T: T.numpy()
        y_test = torch.Tensor(np.ones((1, dim)))

    gp_model.eval()
    # This only works for batched shape now
    # TODO: Make this work again for general multitask kernel
    prior_covar = np.diag(to_numpy(gp_model.covar_module(y_test)).squeeze())

    return prior_covar
