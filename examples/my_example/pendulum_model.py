#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

from acados_template import AcadosModel, AcadosOcp
from casadi import SX, vertcat, sin, cos, Function
import casadi as ca
import numpy as np
from scipy.linalg import block_diag
from casadi_gp_callback import GPDiscreteCallback


def export_pendulum_ode_model(black_box = False) -> AcadosModel:   # inserted parameter l for my simulation

    model_name = 'pendulum'
    if black_box:
        ode_flag = 0
    else:
        ode_flag = 1
    # constants
    m_cart = 1. # mass of the cart [kg]
    m = 0.1 # mass of the ball [kg]
    g = 9.81 # gravity constant [m/s^2]
    l = 0.8 # length of the rod [m]

    # set up states & controls
    x1      = SX.sym('x1')
    theta   = SX.sym('theta')
    v1      = SX.sym('v1')
    dtheta  = SX.sym('dtheta')

    x = vertcat(x1, theta, v1, dtheta)

    F = SX.sym('F')
    u = vertcat(F)

    # xdot
    x1_dot      = SX.sym('x1_dot')
    theta_dot   = SX.sym('theta_dot')
    v1_dot      = SX.sym('v1_dot')
    dtheta_dot  = SX.sym('dtheta_dot')

    xdot = vertcat(x1_dot, theta_dot, ode_flag*v1_dot, ode_flag*dtheta_dot)

    # parameters
    p = []

    # dynamics
    cos_theta = cos(theta)
    sin_theta = sin(theta)
    denominator = m_cart + m - m*cos_theta*cos_theta
    f_expl = vertcat(v1,
                     dtheta,
                     ode_flag*(-m*l*sin_theta*dtheta*dtheta + m*g*cos_theta*sin_theta+F)/denominator,
                     ode_flag*(-m*l*cos_theta*sin_theta*dtheta*dtheta + F*cos_theta+(m_cart+m)*g*sin_theta)/(l*denominator)
                     )

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    # model.z = z
    model.p = p
    model.name = model_name

    # store meta information
    model.x_labels = ['$x$ [m]', r'$\theta$ [rad]', '$v$ [m]', r'$\dot{\theta}$ [rad/s]']
    model.u_labels = ['$F$']
    model.t_label = '$t$ [s]'

    return model


def export_linearized_pendulum(xbar, ubar):
    model = export_pendulum_ode_model()

    val = ca.substitute(ca.substitute(model.f_expl_expr, model.x, xbar), model.u, ubar)
    jac_x = ca.substitute(ca.substitute(ca.jacobian(model.f_expl_expr, model.x), model.x, xbar), model.u, ubar)
    jac_u = ca.substitute(ca.substitute(ca.jacobian(model.f_expl_expr, model.u), model.x, xbar), model.u, ubar)

    model.f_expl_expr = val + jac_x @ (model.x-xbar) + jac_u @ (model.u-ubar)
    model.f_impl_expr = model.f_expl_expr - model.xdot
    model.name += '_linearized'
    return model


def export_pendulum_ode_model_with_discrete_rk4(dT, black_box = False) -> AcadosModel:

    model = export_pendulum_ode_model(black_box)

    x = model.x
    u = model.u

    ode = Function('ode', [x, u], [model.f_expl_expr])
    # set up RK4
    k1 = ode(x,       u)
    k2 = ode(x+dT/2*k1,u)
    k3 = ode(x+dT/2*k2,u)
    k4 = ode(x+dT*k3,  u)
    xf = x + dT/6 * (k1 + 2*k2 + 2*k3 + k4)

    model.disc_dyn_expr = xf
    # print("built RK4 for pendulum model with dT = ", dT)
    # print(xf)
    return model

def export_linearized_pendulum_ode_model_with_discrete_rk4(dT, xbar, ubar):

    model = export_linearized_pendulum(xbar, ubar)

    x = model.x
    u = model.u

    ode = Function('ode', [x, u], [model.f_expl_expr])
    # set up RK4
    k1 = ode(x,       u)
    k2 = ode(x+dT/2*k1,u)
    k3 = ode(x+dT/2*k2,u)
    k4 = ode(x+dT*k3,  u)
    xf = x + dT/6 * (k1 + 2*k2 + 2*k3 + k4)

    model.disc_dyn_expr = xf
    # print("built RK4 for pendulum model with dT = ", dT)
    # print(xf)
    return model

def export_augmented_pendulum_model():
    # pendulum model augmented with algebraic variable just for testing
    model = export_pendulum_ode_model()
    model_name = 'augmented_pendulum'

    z = SX.sym('z', 2, 1)

    f_impl = vertcat( model.xdot - model.f_expl_expr, \
        z - vertcat(model.x[0], model.u**2)
    )

    model.f_impl_expr = f_impl
    model.z = z
    model.name = model_name

    return model

# Till here the code is the same as in the original file
# ----------------------------------------------------------------------------------

def export_pendulum_ode_real_model() -> AcadosModel:  # same model as before, just a different pendulum length

    model_name = 'real_pendulum'

    # constants
    m_cart = 1. # mass of the cart [kg]
    m = 0.1 # mass of the ball [kg]
    g = 9.81 # gravity constant [m/s^2]
    l = 0.5 # length of the rod [m]

    # set up states & controls
    x1      = SX.sym('x1')
    theta   = SX.sym('theta')
    v1      = SX.sym('v1')
    dtheta  = SX.sym('dtheta')

    x = vertcat(x1, theta, v1, dtheta)

    F = SX.sym('F')
    u = vertcat(F)

    # xdot
    x1_dot      = SX.sym('x1_dot')
    theta_dot   = SX.sym('theta_dot')
    v1_dot      = SX.sym('v1_dot')
    dtheta_dot  = SX.sym('dtheta_dot')

    xdot = vertcat(x1_dot, theta_dot, v1_dot, dtheta_dot)

    # parameters
    p = []

    # dynamics
    cos_theta = cos(theta)
    sin_theta = sin(theta)
    denominator = m_cart + m - m*cos_theta*cos_theta
    f_expl = vertcat(v1,
                     dtheta,
                     (-m*l*sin_theta*dtheta*dtheta + m*g*cos_theta*sin_theta+F)/denominator,
                     (-m*l*cos_theta*sin_theta*dtheta*dtheta + F*cos_theta+(m_cart+m)*g*sin_theta)/(l*denominator)
                     )

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    # model.z = z
    model.p = p
    model.name = model_name

    # store meta information
    model.x_labels = ['$x$ [m]', r'$\theta$ [rad]', '$v$ [m]', r'$\dot{\theta}$ [rad/s]']
    model.u_labels = ['$F$']
    model.t_label = '$t$ [s]'

    return model


def export_ocp_cartpendulum_discrete(N, T, only_lower_bounds=False, **model_kwargs):
    # initial state: cart at 0, pendulum upright
    x0 = np.array([0.0, np.pi, 0.0, 0.0])

    # input bounds: force
    lb_u = -50.0
    ub_u = 50.0

    # cost weights
    Q = np.diagflat([15.0, 10.0, 0.1, 0.1])  # [cart, theta, cart_vel, omega]
    R = np.array([[0.1]])                    # [u]
    Qe = np.diagflat([10.0, 10.0, 0.1, 0.1])  # terminal cost

    # load the discrete-time model
    dt = T / N
    model = export_pendulum_ode_model_with_discrete_rk4(dt)

    # define OCP
    ocp = AcadosOcp()
    ocp.model = model
    ocp.dims.N = N

    nx = model.x.shape[0]
    nu = model.u.shape[0]
    ny = nx + nu
    ny_e = nx

    ocp.dims.nx = nx
    ocp.dims.nu = nu
    ocp.dims.np = model.p.shape[0] if hasattr(model, "p") and isinstance(model.p, SX) else 0

    # cost
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.W = block_diag(Q, R)
    ocp.cost.W_0 = ocp.cost.W
    ocp.cost.W_e = Qe

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)

    ocp.cost.Vu = np.zeros((ny, nu))
    ocp.cost.Vu[nx:, :] = np.eye(nu)

    ocp.cost.Vx_e = np.eye(ny_e)
    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))

    # constraints
    ocp.constraints.constr_type = "BGH"
    ocp.constraints.lbu = np.array([lb_u])
    ocp.constraints.ubu = np.array([ub_u])
    ocp.constraints.idxbu = np.array(range(nu))
    ocp.constraints.x0 = x0

    # nonlinear constraints
    if hasattr(model, "con_h_expr") and model.con_h_expr is not None:
        nh = model.con_h_expr.shape[0]
        ocp.dims.nh = nh
        ocp.model.con_h_expr = model.con_h_expr

        inf = 1e6
        if only_lower_bounds:
            ocp.constraints.lh = np.zeros((nh,))
            ocp.constraints.uh = np.full((nh,), inf)
        else:
            ocp.constraints.lh = -np.ones((nh,)) * inf  # or your actual lower bounds
            ocp.constraints.uh = np.ones((nh,)) * inf   # or your actual upper bounds
    else:
        ocp.dims.nh = 0
        # Do NOT set lh, uh, or con_h_expr if nh == 0

    # solver options
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"  # ‘PARTIAL_CONDENSING_HPIPM’, ‘FULL_CONDENSING_QPOASES’, 
    #‘FULL_CONDENSING_HPIPM’, ‘PARTIAL_CONDENSING_QPDUNES’, ‘PARTIAL_CONDENSING_OSQP’, ‘FULL_CONDENSING_DAQP’
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.tf = T
    ocp.solver_options.tol = 1e-2

    return ocp


def export_pendulum_ode_real_model_with_discrete_rk4(dT):

    model = export_pendulum_ode_real_model()

    x = model.x
    u = model.u

    ode = Function('ode', [x, u], [model.f_expl_expr])
    # set up RK4
    k1 = ode(x,       u)
    k2 = ode(x+dT/2*k1,u)
    k3 = ode(x+dT/2*k2,u)
    k4 = ode(x+dT*k3,  u)
    xf = x + dT/6 * (k1 + 2*k2 + 2*k3 + k4)

    model.disc_dyn_expr = xf
    # print("built RK4 for pendulum model with dT = ", dT)
    # print(xf)
    return model

def export_augmented_pendulum_ode_model(dt, black_box=False) -> AcadosModel:
    """
    Returns an augmented pendulum ODE model with memory of 2 previous time steps.
    
    Args:
        dt (float): Time step for discretization.
        black_box (bool): If True, disables ODE dynamics (used for learning-based models).
    
    Returns:
        AcadosModel: Configured model for use with acados.
    """

    model_name = 'augmented_pendulum'
    ode_flag = 0 if black_box else 1

    # Constants
    m_cart = 1.0
    m = 0.1
    g = 9.81
    l = 0.8

    # Current state
    x1 = SX.sym('x')
    x2 = SX.sym('theta')
    x3 = SX.sym('v')
    x4 = SX.sym('omega')

    # Previous states
    x1_p1 = SX.sym('x_p1')
    x2_p1 = SX.sym('theta_p1')
    x3_p1 = SX.sym('v_p1')
    x4_p1 = SX.sym('omega_p1')

    x1_p2 = SX.sym('x_p2')
    x2_p2 = SX.sym('theta_p2')
    x3_p2 = SX.sym('v_p2')
    x4_p2 = SX.sym('omega_p2')

    # Input
    F = SX.sym('F')
    u = vertcat(F)

    # Compose full state vector
    x_curr = vertcat(x1, x2, x3, x4)
    x_prev1 = vertcat(x1_p1, x2_p1, x3_p1, x4_p1)
    x_prev2 = vertcat(x1_p2, x2_p2, x3_p2, x4_p2)
    x_full = vertcat(x_curr, x_prev1, x_prev2)

    # xdot
    x1_dot     = SX.sym('x1_dot')
    theta_dot  = SX.sym('theta_dot')
    v1_dot     = SX.sym('v1_dot')
    dtheta_dot = SX.sym('dtheta_dot')
    xdot = vertcat(x1_dot, theta_dot, ode_flag*v1_dot, ode_flag*dtheta_dot)

    # No parameters
    p = []

    # Explicit dynamics
    cos_theta = cos(x2)
    sin_theta = sin(x2)
    denominator = m_cart + m - m * cos_theta**2
    f_expl = vertcat(
        x3,
        x4,
        ode_flag * (-m * l * sin_theta * x4**2 + m * g * cos_theta * sin_theta + F) / denominator,
        ode_flag * (-m * l * cos_theta * sin_theta * x4**2 + F * cos_theta + (m_cart + m) * g * sin_theta) / (l * denominator)
    )

    # Implicit dynamics
    f_impl = xdot - f_expl

    # Discretized Euler dynamics
    x_next = x_curr + dt * f_expl
    x_next_full = vertcat(x_next, x_curr, x_prev1)

    # Build acados model
    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x_full
    model.xdot = xdot
    model.u = u
    model.p = p
    model.name = model_name
    model.disc_dyn_expr = x_next_full

    # Meta
    model.x_labels = ['$x$ [m]', r'$\theta$ [rad]', '$v$ [m/s]', r'$\dot{\theta}$ [rad/s]']
    model.u_labels = ['$F$']
    model.t_label = '$t$ [s]'

    return model

def export_augmented_pendulum_model_with_rk4(dt, black_box=False):

    model = export_augmented_pendulum_ode_model(dt, black_box)

    # Extract state vectors
    x_full = model.x        # full state: [x_k, x_{k-1}, x_{k-2}]
    u = model.u

    # Split current, past1, past2 from x_full
    x_curr = x_full[0:4]    # x_k
    x_prev1 = x_full[4:8]   # x_{k-1}
    x_prev2 = x_full[8:12]  # x_{k-2}

    # Build explicit ODE function using current state and input
    ode = Function('ode_aug', [x_curr, u], [model.f_expl_expr])

    # --- RK4 integration on current state ---
    k1 = ode(x_curr,           u)
    k2 = ode(x_curr + dt/2*k1, u)
    k3 = ode(x_curr + dt/2*k2, u)
    k4 = ode(x_curr + dt*k3,   u)
    x_next = x_curr + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    # Construct next full state: shift history
    x_next_full = vertcat(x_next, x_curr, x_prev1)

    # Assign to model
    model.disc_dyn_expr = x_next_full

    return model

def export_augmented_pendulum_ode_real_model(dt, black_box=False) -> AcadosModel:
    """
    Returns an augmented pendulum ODE model with memory of 2 previous time steps.
    
    Args:
        dt (float): Time step for discretization.
        black_box (bool): If True, disables ODE dynamics (used for learning-based models).
    
    Returns:
        AcadosModel: Configured model for use with acados.
    """

    model_name = 'augmented_pendulum'
    ode_flag = 0 if black_box else 1

    # Constants
    m_cart = 1.0
    m = 0.1
    g = 9.81
    l = 0.5

    # Current state
    x1 = SX.sym('x')
    x2 = SX.sym('theta')
    x3 = SX.sym('v')
    x4 = SX.sym('omega')

    # Previous states
    x1_p1 = SX.sym('x_p1')
    x2_p1 = SX.sym('theta_p1')
    x3_p1 = SX.sym('v_p1')
    x4_p1 = SX.sym('omega_p1')

    x1_p2 = SX.sym('x_p2')
    x2_p2 = SX.sym('theta_p2')
    x3_p2 = SX.sym('v_p2')
    x4_p2 = SX.sym('omega_p2')

    # Input
    F = SX.sym('F')
    u = vertcat(F)

    # Compose full state vector
    x_curr = vertcat(x1, x2, x3, x4)
    x_prev1 = vertcat(x1_p1, x2_p1, x3_p1, x4_p1)
    x_prev2 = vertcat(x1_p2, x2_p2, x3_p2, x4_p2)
    x_full = vertcat(x_curr, x_prev1, x_prev2)

    # xdot
    x1_dot     = SX.sym('x1_dot')
    theta_dot  = SX.sym('theta_dot')
    v1_dot     = SX.sym('v1_dot')
    dtheta_dot = SX.sym('dtheta_dot')
    xdot = vertcat(x1_dot, theta_dot, ode_flag*v1_dot, ode_flag*dtheta_dot)

    # No parameters
    p = []

    # Explicit dynamics
    cos_theta = cos(x2)
    sin_theta = sin(x2)
    denominator = m_cart + m - m * cos_theta**2
    f_expl = vertcat(
        x3,
        x4,
        ode_flag * (-m * l * sin_theta * x4**2 + m * g * cos_theta * sin_theta + F) / denominator,
        ode_flag * (-m * l * cos_theta * sin_theta * x4**2 + F * cos_theta + (m_cart + m) * g * sin_theta) / (l * denominator)
    )

    # Implicit dynamics
    f_impl = xdot - f_expl

    # Discretized Euler dynamics
    x_next = x_curr + dt * f_expl
    x_next_full = vertcat(x_next, x_curr, x_prev1)

    # Build acados model
    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x_full
    model.xdot = xdot
    model.u = u
    model.p = p
    model.name = model_name
    model.disc_dyn_expr = x_next_full

    # Meta
    model.x_labels = ['$x$ [m]', r'$\theta$ [rad]', '$v$ [m/s]', r'$\dot{\theta}$ [rad/s]']
    model.u_labels = ['$F$']
    model.t_label = '$t$ [s]'

    return model
#def export_discrete_gp_blackbox_model(gp_model, nx, nu):
#    
#    x = SX.sym("x", nx)
#    u = SX.sym("u", nu)
#
#    gp_callback = GPDiscreteCallback(gp_model, nx, nu)
#
#    model = AcadosModel()
#    model.name = "gp_black_box"
#    model.x = x
#    model.u = u
#    model.dim_nx = nx
#    model.dim_nu = nu
#    model.dyn_ext_fun = gp_callback
#    model.model_type = "external"
#    model.dyn_type = "discrete"
#
#    # VERY IMPORTANT: disable continuous dynamics
#    model.f_expl_expr = None
#    model.f_impl_expr = None
#
#    model.cost_y_expr = vertcat(model.x, model.u)
#    model.cost_y_expr_e = model.x
#    model.cost_y_expr_0 = model.cost_y_expr
#
#    return model

#def export_ocp_blackbox_discrete(gp_model, N, T, x0=None, only_lower_bounds=False):
#    """
#    Create an Acados OCP using a black-box discrete CasADi model (like a GP model).
#    
#    Inputs:
#        - gp_model: your trained GPyTorch GP model
#        - N: number of shooting intervals
#        - T: time horizon [s]
#        - x0: initial condition (default [0, pi, 0, 0])
#        - only_lower_bounds: if True, set only lower bounds on h (if present)
#        
#    Returns:
#        - AcadosOcp object
#    """
#    
#    from casadi_gp_callback import GPDiscreteCallback  # Assuming your Callback is in this file!
#
#    # system dimensions
#    nx = 4  # adjust if different
#    nu = 1  # adjust if different
#
#    # initial condition
#    if x0 is None:
#        x0 = np.array([0.0, np.pi, 0.0, 0.0])  # cart-pendulum upright
#
#    # input bounds
#    lb_u = -50.0
#    ub_u = 50.0
#
#    # cost weights
#    Q = np.diagflat([10.0, 10.0, 0.1, 0.1])  # [cart, theta, cart_vel, omega]
#    R = np.array([[0.1]])                    # [u]
#    Qe = np.diagflat([10.0, 10.0, 0.1, 0.1])  # terminal cost
#
#    # create black-box model
#    model = AcadosModel()
#    model.name = "cartpole_gp"
#    model.x = SX.sym('x', nx)
#    model.u = SX.sym('u', nu)
#    model.xdot = SX.sym('xdot', nx)
#    # model.p = SX.sym('p', 0)  # no parameters
#    # model.z = SX.sym('z', 0)  # no algebraic variables
#
#    # Dummy continuous dynamics (returns zero)
#    zero_rhs = ca.SX.zeros(nx)
#    model.f_impl_expr = zero_rhs
#    model.f_expl_expr = zero_rhs
#    
#    model.dyn_expr_f = ca.Function("f_expl", [model.x, model.u], [zero_rhs])
#    model.dyn_expr_f_impl = ca.Function("f_impl", [model.x, model.u, model.xdot], [zero_rhs])
#
#
#    # set the Callback
#    gp_callback = GPDiscreteCallback(gp_model, nx=nx, nu=nu)
#    model.disc_dyn_ext_fun_type = 'generic'
#    # model.disc_dyn_ext_fun = AcadosExternalFunction()
#    model.dyn_type = "discrete"
#    
#    # define ocp
#    ocp = AcadosOcp()
#    ocp.model = model
#    ocp.model.dyn_type = "discrete"
#    ocp.solver_options.integrator_type = "DISCRETE"
#    ocp.dims.N = N
#    ocp.dims.nx = nx
#    ocp.dims.nu = nu
#    ocp.dims.np = 0
#    ocp.dims.nz = 0
#
#    # Cost
#    ny = nx + nu
#    ny_e = nx
#
#    ocp.cost.cost_type = "NONLINEAR_LS"
#    ocp.cost.cost_type_e = "NONLINEAR_LS"
#
#    # Weights
#    ocp.cost.W = np.eye(ny)
#    ocp.cost.W_e = np.eye(ny_e)
#    ocp.cost.W_0 = np.eye(ny)  # <-- needed for stage 0
#
#    # References
#    ocp.cost.yref = np.zeros((ny,))
#    ocp.cost.yref_e = np.zeros((ny_e,))
#    ocp.cost.yref_0 = np.zeros((ny,))  # <-- needed for stage 0
#
#
#    # Dimensions
#    ocp.dims.ny = ny
#    ocp.dims.ny_e = ny_e
#    ocp.dims.ny_0 = ny  # <-- needed for stage 0
#
#    # ocp.cost.W = block_diag(Q, R)
#    # ocp.cost.W_0 = ocp.cost.W
#    # ocp.cost.W_e = Qe
#
#    # ocp.cost.Vx = np.zeros((ny, nx))
#    # ocp.cost.Vx[:nx, :nx] = np.eye(nx)
#
#    # ocp.cost.Vu = np.zeros((ny, nu))
#    # ocp.cost.Vu[nx:, :] = np.eye(nu)
#
#    # ocp.cost.Vx_e = np.eye(ny_e)
#    # ocp.cost.yref = np.zeros((ny,))
#    # ocp.cost.yref_e = np.zeros((ny_e,))
#
#    # Constraints
#    ocp.constraints.constr_type = "BGH"
#    ocp.constraints.lbu = np.array([lb_u])
#    ocp.constraints.ubu = np.array([ub_u])
#    ocp.constraints.idxbu = np.array(range(nu))
#    ocp.constraints.x0 = x0
#
#    # No nonlinear constraints (h(x,u)) in this example
#    ocp.dims.nh = 0
#
#    # Solver options
#    ocp.solver_options.integrator_type = "DISCRETE"
#    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
#    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
#    ocp.solver_options.nlp_solver_type = "SQP_RTI"
#    ocp.solver_options.tf = T
#    ocp.solver_options.tol = 1e-2
#
#    return ocp
#