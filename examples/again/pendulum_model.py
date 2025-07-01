from acados_template import AcadosModel, AcadosOcp
from casadi import SX, vertcat, sin, cos, Function
import casadi as ca
import numpy as np
from scipy.linalg import block_diag

from casadi import SX, vertcat, sin, cos
from acados_template import AcadosModel

def export_pendulum_ode_model_actuation(dT, black_box = False) -> AcadosModel:
    model_name = 'pendulum'
    ode_flag = 0 if black_box else 1

    # constants
    m_cart = 1.0   # mass of the cart [kg]
    m = 0.1        # mass of the ball [kg]
    g = 9.81       # gravity constant [m/s^2]

    # pendulum length as parameter
    l_param = SX.sym('l')  # symbolic parameter
    tau     = SX.sym('tau')  # time constant for control input

    # set up states & controls
    x1      = SX.sym('x1')
    theta   = SX.sym('theta')
    v1      = SX.sym('v1')
    dtheta  = SX.sym('dtheta')
    u_act   = SX.sym('u_act')

    x = vertcat(x1, theta, v1, dtheta, u_act)

    F = SX.sym('F')
    u = vertcat(F)

    # xdot
    x1_dot     = SX.sym('x1_dot')
    theta_dot  = SX.sym('theta_dot')
    v1_dot     = SX.sym('v1_dot')
    dtheta_dot = SX.sym('dtheta_dot')
    u_act_dot  = SX.sym('u_act_dot')

    xdot = vertcat(x1_dot, theta_dot, v1_dot, dtheta_dot, u_act_dot)

    # parameters vector (only l for now)
    p = vertcat(l_param, tau)

    # dynamics
    cos_theta = cos(theta)
    sin_theta = sin(theta)
    denominator = m_cart + m - m * cos_theta**2

    f_expl = vertcat(
        v1,
        dtheta,
        ode_flag*(-m * l_param * sin_theta * dtheta**2 + m * g * cos_theta * sin_theta + u_act) / denominator,
        ode_flag*(-m * l_param * cos_theta * sin_theta * dtheta**2 + u_act * cos_theta + (m_cart + m) * g * sin_theta) / (l_param * denominator),
        (1 / tau ) * (u - u_act),  # u_act is the previous control input
    )

    f_impl = xdot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.p = p
    model.name = model_name

    ode = Function('ode', [x, u, p], [f_expl])
    # set up RK4
    k1 = ode(x,        u, p)
    k2 = ode(x+dT/2*k1,u, p)
    k3 = ode(x+dT/2*k2,u, p)
    k4 = ode(x+dT*k3,  u, p)
    xf = x + dT/6 * (k1 + 2*k2 + 2*k3 + k4)

    model.disc_dyn_expr = xf

    # Meta info
    model.x_labels = ['$x$ [m]', r'$\theta$ [rad]', '$v$ [m/s]', r'$\dot{\theta}$ [rad/s]']
    model.u_labels = ['$F$']
    model.t_label = '$t$ [s]'

    return model

def export_pendulum_ode_model() -> AcadosModel:
    model_name = 'pendulum'

    # constants
    m_cart = 1.0   # mass of the cart [kg]
    m = 0.1        # mass of the ball [kg]
    g = 9.81       # gravity constant [m/s^2]

    # pendulum length as parameter
    l_param = SX.sym('l')  # symbolic parameter

    # set up states & controls
    x1      = SX.sym('x1')
    theta   = SX.sym('theta')
    v1      = SX.sym('v1')
    dtheta  = SX.sym('dtheta')

    x = vertcat(x1, theta, v1, dtheta)

    F = SX.sym('F')
    u = vertcat(F)

    # xdot
    x1_dot     = SX.sym('x1_dot')
    theta_dot  = SX.sym('theta_dot')
    v1_dot     = SX.sym('v1_dot')
    dtheta_dot = SX.sym('dtheta_dot')

    xdot = vertcat(x1_dot, theta_dot, v1_dot, dtheta_dot)

    # parameters vector (only l for now)
    p = vertcat(l_param)

    # dynamics
    cos_theta = cos(theta)
    sin_theta = sin(theta)
    denominator = m_cart + m - m * cos_theta**2

    f_expl = vertcat(
        v1,
        dtheta,
        (-m * p * sin_theta * dtheta**2 + m * g * cos_theta * sin_theta + u) / denominator,
        (-m * p * cos_theta * sin_theta * dtheta**2 + u * cos_theta + (m_cart + m) * g * sin_theta) / (p * denominator),
    )

    f_impl = xdot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.p = p
    model.name = model_name

    # Meta info
    model.x_labels = ['$x$ [m]', r'$\theta$ [rad]', '$v$ [m/s]', r'$\dot{\theta}$ [rad/s]']
    model.u_labels = ['$F$']
    model.t_label = '$t$ [s]'

    return model

def export_pendulum_ode_model_double_delay(dT, black_box=False) -> AcadosModel:
    '''
    Define model for the pendulum with both dynamic actuation delay, and constant input delay of one step
    '''
    model_name = 'pendulum'
    ode_flag = 0 if black_box else 1

    # constants
    m_cart = 1.0   # mass of the cart [kg]
    m = 0.1        # mass of the ball [kg]
    g = 9.81       # gravity constant [m/s^2]

    # pendulum length as parameter
    l_param = SX.sym('l')  # symbolic parameter
    tau     = SX.sym('tau')  # time constant for control input

    # set up states & controls
    x1      = SX.sym('x1')
    theta   = SX.sym('theta')
    v1      = SX.sym('v1')
    dtheta  = SX.sym('dtheta')
    u_act   = SX.sym('u_act')
    u_past = SX.sym('u_past')

    x_curr = vertcat(x1, theta, v1, dtheta, u_act)

    x_full = vertcat(x_curr, u_past)

    F = SX.sym('F')
    u = vertcat(F)

    # xdot
    x1_dot     = SX.sym('x1_dot')
    theta_dot  = SX.sym('theta_dot')
    v1_dot     = SX.sym('v1_dot')
    dtheta_dot = SX.sym('dtheta_dot')
    u_act_dot  = SX.sym('u_act_dot')
    u_past_dot = SX.sym('u_past_dot')

    xdot = vertcat(x1_dot, theta_dot, v1_dot, dtheta_dot, u_act_dot, u_past_dot)

    # parameters vector (only l for now)
    p = vertcat(l_param, tau)

    # dynamics
    cos_theta = cos(theta)
    sin_theta = sin(theta)
    denominator = m_cart + m - m * cos_theta**2

    f_expl = vertcat(
        v1,
        dtheta,
        ode_flag*(-m * l_param * sin_theta * dtheta**2 + m * g * cos_theta * sin_theta + u_act) / denominator,
        ode_flag*(-m * l_param * cos_theta * sin_theta * dtheta**2 + u_act * cos_theta + (m_cart + m) * g * sin_theta) / (l_param * denominator),
        (1 / tau ) * (u_past - u_act),  # u_act is the previous control input
        (u - u_past) / dT
    )

    cont_dynamics = vertcat(
        v1,
        dtheta,
        ode_flag*(-m * l_param * sin_theta * dtheta**2 + m * g * cos_theta * sin_theta + u_act) / denominator,
        ode_flag*(-m * l_param * cos_theta * sin_theta * dtheta**2 + u_act * cos_theta + (m_cart + m) * g * sin_theta) / (l_param * denominator),
        (1 / tau ) * (u_past - u_act),  # u_act is the previous control input
    )

    f_impl = xdot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x_full
    model.xdot = xdot
    model.u = u
    model.p = p
    model.name = model_name

    ode = Function('ode', [x_curr, u, u_past, p], [cont_dynamics])
    # set up RK4
    k1 = ode(x_curr,        u, u_past, p)
    k2 = ode(x_curr+dT/2*k1,u, u_past, p)
    k3 = ode(x_curr+dT/2*k2,u, u_past, p)
    k4 = ode(x_curr+dT*k3,  u, u_past, p)
    xf = x_curr + dT/6 * (k1 + 2*k2 + 2*k3 + k4)

    model.disc_dyn_expr = vertcat(xf, u)

    # Meta info
    model.x_labels = ['$x$ [m]', r'$\theta$ [rad]', '$v$ [m/s]', r'$\dot{\theta}$ [rad/s]']
    model.u_labels = ['$F$']
    model.t_label = '$t$ [s]'

    return model

#def export_linearized_pendulum(xbar, ubar):
#    model = export_pendulum_ode_model()
#
#    val = ca.substitute(ca.substitute(model.f_expl_expr, model.x, xbar), model.u, ubar)
#    jac_x = ca.substitute(ca.substitute(ca.jacobian(model.f_expl_expr, model.x), model.x, xbar), model.u, ubar)
#    jac_u = ca.substitute(ca.substitute(ca.jacobian(model.f_expl_expr, model.u), model.x, xbar), model.u, ubar)
#
#    model.f_expl_expr = val + jac_x @ (model.x-xbar) + jac_u @ (model.u-ubar)
#    model.f_impl_expr = model.f_expl_expr - model.xdot
#    model.name += '_linearized'
#    return model

def export_discrete_pendulum_ode_model(dT, black_box = False) -> AcadosModel:

    model_name = 'pendulum'
    if black_box:
        ode_flag = 0    # flag to disable ODE dynamics for black-box models
    else:
        ode_flag = 1    # flag to enable ODE dynamics for physical models

    # constants
    m_cart = 1.0   # mass of the cart [kg]
    m = 0.1        # mass of the ball [kg]
    g = 9.81       # gravity constant [m/s^2]

    # pendulum length as parameter
    l_param = SX.sym('l')  # symbolic parameter

    # set up states & controls
    x1      = SX.sym('x1')
    x2   = SX.sym('theta')
    x3      = SX.sym('v1')
    x4  = SX.sym('dtheta')

    x = vertcat(x1, x2, x3, x4)

    F = SX.sym('F')
    u = vertcat(F)

    # xdot
    x1_dot     = SX.sym('x1_dot')
    x2_dot  = SX.sym('theta_dot')
    x3_dot     = SX.sym('v1_dot')
    x4_dot = SX.sym('dtheta_dot')

    xdot = vertcat(x1_dot, x2_dot, x3_dot, x4_dot)

    # parameters vector (only l for now)
    p = vertcat(l_param)

    # dynamics
    cos_theta = cos(x2)
    sin_theta = sin(x2)
    denominator = m_cart + m - m * cos_theta**2

    f_expl = vertcat(
        x3,
        x4,
        ode_flag*(-m * p * sin_theta * x4**2 + m * g * cos_theta * sin_theta + u) / denominator,
        ode_flag*(-m * p * cos_theta * sin_theta * x4**2 + u * cos_theta + (m_cart + m) * g * sin_theta) / (p * denominator),
    )

    f_impl = xdot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.p = p
    model.name = model_name

    ode = Function('ode', [x, u, p], [f_expl])
    # set up RK4
    k1 = ode(x,        u, p)
    k2 = ode(x+dT/2*k1,u, p)
    k3 = ode(x+dT/2*k2,u, p)
    k4 = ode(x+dT*k3,  u, p)
    xf = x + dT/6 * (k1 + 2*k2 + 2*k3 + k4)

    model.disc_dyn_expr = xf

    # print("built RK4 for pendulum model with dT = ", dT)
    # print(xf)
    return model

def export_discrete_integrator(model):

    # Compute linearization matrices around (xbar, ubar)
    A_sym = ca.jacobian(model.disc_dyn_expr, model.x)
    B_sym = ca.jacobian(model.disc_dyn_expr, model.u)

    A_fun = Function('A_fun', [model.x, model.u, model.p], [A_sym])
    B_fun = Function('B_fun', [model.x, model.u, model.p], [B_sym])

    x_next_lin = A_sym @ model.x + B_sym @ model.u

    # CasADi function: takes (x, u, p), returns x_next
    linearized_dynamics_fun = ca.Function('linear_dynamics_fun', [model.x, model.u, model.p], [x_next_lin])

    discrete_dynamics_fun = ca.Function("discret_dyn_fun", [model.x, model.u, model.p], [model.disc_dyn_expr])

    return linearized_dynamics_fun, discrete_dynamics_fun

#def export_linearized_pendulum_ode_model_with_discrete_rk4(dT, xbar, ubar):
#
#    model = export_linearized_pendulum(xbar, ubar)
#
#    x = model.x
#    u = model.u
#
#    ode = Function('ode', [x, u], [model.f_expl_expr])
#    # set up RK4
#    k1 = ode(x,       u)
#    k2 = ode(x+dT/2*k1,u)
#    k3 = ode(x+dT/2*k2,u)
#    k4 = ode(x+dT*k3,  u)
#    xf = x + dT/6 * (k1 + 2*k2 + 2*k3 + k4)
#
#    model.disc_dyn_expr = xf
#    # print("built RK4 for pendulum model with dT = ", dT)
#    # print(xf)
#    return model

#def export_augmented_pendulum_model():
#    # pendulum model augmented with algebraic variable just for testing
#    model = export_pendulum_ode_model()
#    model_name = 'augmented_pendulum'
#
#    z = SX.sym('z', 2, 1)
#
#    f_impl = vertcat( model.xdot - model.f_expl_expr, \
#        z - vertcat(model.x[0], model.u**2)
#    )
#
#    model.f_impl_expr = f_impl
#    model.z = z
#    model.name = model_name
#
#    return model
#############################################################################################
#############################################################################################

#############################################################################################
#############################################################################################

def discrete_pendulum_ode_model_with_past_states_and_input_delay(dt, black_box=False) -> AcadosModel:
    '''
    Discrete dynamics with constant input delay of two steps. Model includes past states and inputs.
    '''

    model_name = 'pendulum_discrete'
    ode_flag = 0 if black_box else 1
    

    # Constants
    m_cart = 1.0
    m = 0.1
    g = 9.81
    # l = 0.8

    # Current state
    x1 = SX.sym('x')        # cart position
    x2 = SX.sym('theta')    # pendulum angle
    x3 = SX.sym('v')        # cart velocity
    x4 = SX.sym('omega')    # pendulum angular velocity
    # Previous states (x_{k-1})
    x1_p1 = SX.sym('x_p1')
    x2_p1 = SX.sym('theta_p1')
    u_p1 = SX.sym('u_p1')  # control input at previous time step
    # Previous states (x_{k-2})
    x1_p2 = SX.sym('x_p2')
    x2_p2 = SX.sym('theta_p2')
    u_p2 = SX.sym('u_p2')  # control input at previous time step

    # Full state
    x_curr = vertcat(x1, x2, x3, x4)
    x_prev1 = vertcat(x1_p1, x2_p1, u_p1)
    x_prev2 = vertcat(x1_p2, x2_p2, u_p2)
    x_full = vertcat(x_curr, x_prev1, x_prev2)

    # Control input
    F = SX.sym('F')
    u = vertcat(F)

    # Parameters (e.g., black-box parameter p)
    p = SX.sym('p')
    l = vertcat(p)

    # Define xdot (12D, matching state)
    xdot_curr = SX.sym('xdot_curr', 4)
    xdot_prev1 = SX.sym('xdot_prev1', 2)
    udot_prev1 = SX.sym('udot_prev1', 1)
    xdot_prev2 = SX.sym('xdot_prev2', 2)
    udot_prev2 = SX.sym('udot_prev2', 1)
    xdot = vertcat(xdot_curr, xdot_prev1, udot_prev1, xdot_prev2, udot_prev2)

    # Flag for enabling/disabling physical dynamics
    ode_flag = 0 if black_box else 1

    # --- Physical Dynamics (only for current state) ---
    cos_theta = cos(x2)
    sin_theta = sin(x2)
    denominator = m_cart + m - m * cos_theta**2

    # --- Full Explicit Dynamics ---
    f_expl = vertcat(
        ode_flag * x3,
        ode_flag * x4,
        ode_flag * (-m * l * sin_theta * x4**2 + m * g * cos_theta * sin_theta + u_p2) / denominator,
        ode_flag * (-m * l * cos_theta * sin_theta * x4**2 + u_p2 * cos_theta + (m_cart + m) * g * sin_theta) / (l * denominator),
        (x1 - x1_p1) / dt,
        (x2 - x2_p1) / dt,
        (u - u_p1) / dt,
        (x1_p1 - x1_p2) / dt,
        (x2_p1 - x2_p2) / dt, 
        (u_p1 - u_p2) / dt,
    )

    cont_dyn = vertcat(
        x3,
        x4,
        ode_flag * (-m * l * sin_theta * x4**2 + m * g * cos_theta * sin_theta + u_p2) / denominator,
        ode_flag * (-m * l * cos_theta * sin_theta * x4**2 + u_p2 * cos_theta + (m_cart + m) * g * sin_theta) / (l * denominator),  
    )

    # --- Implicit Dynamics ---
    f_impl = xdot - f_expl

    # --- Assemble Acados Model ---
    model = AcadosModel()
    model.name = model_name
    model.x = x_full
    model.xdot = xdot
    model.u = u
    model.p = p
    # model.f_expl = f_expl
    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl
    # model.disc_dyn_expr = x_next_full  # fallback Euler discretization

    ode = Function('ode', [x_curr, u_p2, p], [cont_dyn])
    # set up RK4
    k1 = ode(x_curr,        u_p2, p)
    k2 = ode(x_curr+dt/2*k1,u_p2, p)
    k3 = ode(x_curr+dt/2*k2,u_p2, p)
    k4 = ode(x_curr+dt*k3,  u_p2, p)
    xf = x_curr + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    full_discrete = vertcat(
        xf,    # current state update (4 states), based on RK4 with previous control input (u_p2)
        x1,    # previous state x1_p1 is just the current x1
        x2,     # previous state x2_p1 is just the current x2
        u,     # previous control input u_p1 is just the current u
        x1_p1, 
        x2_p1, 
        u_p1
    )

    model.disc_dyn_expr = full_discrete  # full discrete dynamics with past states
    # Optional labels
    model.x_labels = ['$x$', r'$\theta$', '$\dot{x}$', r'$\dot{\theta}$',
                      '$x_{-1}$', r'$\theta_{-1}$', '$\dot{x}_{-1}$', r'$\dot{\theta}_{-1}$',
                      '$x_{-2}$', r'$\theta_{-2}$', '$\dot{x}_{-2}$', r'$\dot{\theta}_{-2}$']
    model.u_labels = ['$F$']
    model.t_label = '$t$ [s]'

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

    # Constants
    m_cart = 1.0
    m = 0.1
    g = 9.81
    # l = 0.8

    # Current state
    x1 = SX.sym('x')        # cart position
    x2 = SX.sym('theta')    # pendulum angle
    x3 = SX.sym('v')        # cart velocity
    x4 = SX.sym('omega')    # pendulum angular velocity
    # Previous states (x_{k-1})
    x1_p1 = SX.sym('x_p1')
    x2_p1 = SX.sym('theta_p1')
    u_p1 = SX.sym('u_p1')  # control input at previous time step
    # Previous states (x_{k-2})
    x1_p2 = SX.sym('x_p2')
    x2_p2 = SX.sym('theta_p2')
    u_p2 = SX.sym('u_p2')  # control input at previous time step

    # Full state
    x_curr = vertcat(x1, x2, x3, x4)
    x_prev1 = vertcat(x1_p1, x2_p1, u_p1)
    x_prev2 = vertcat(x1_p2, x2_p2, u_p2)
    x_full = vertcat(x_curr, x_prev1, x_prev2)

    # Control input
    F = SX.sym('F')
    u = vertcat(F)

    # Parameters (e.g., black-box parameter p)
    p = SX.sym('p')
    l = vertcat(p)

    # Define xdot (12D, matching state)
    xdot_curr = SX.sym('xdot_curr', 4)
    xdot_prev1 = SX.sym('xdot_prev1', 2)
    u_prev1 = SX.sym('u_prev1', 1)
    xdot_prev2 = SX.sym('xdot_prev2', 2)
    u_prev2 = SX.sym('u_prev2', 1)
    xdot = vertcat(xdot_curr, xdot_prev1, u_prev1, xdot_prev2, u_prev2)

    # Flag for enabling/disabling physical dynamics
    ode_flag = 0 if black_box else 1

    # --- Physical Dynamics (only for current state) ---
    cos_theta = cos(x2)
    sin_theta = sin(x2)
    denominator = m_cart + m - m * cos_theta**2

    # --- Full Explicit Dynamics ---
    f_expl = vertcat(
        ode_flag * x3,
        ode_flag * x4,
        ode_flag * (-m * p * sin_theta * x4**2 + m * g * cos_theta * sin_theta + F) / denominator,
        ode_flag * (-m * p * cos_theta * sin_theta * x4**2 + F * cos_theta + (m_cart + m) * g * sin_theta) / (p * denominator),
        (x1 - x1_p1) / dt,
        (x2 - x2_p1) / dt,
        (u - u_p1) / dt,
        (x1_p1 - x1_p2) / dt,
        (x2_p1 - x2_p2) / dt, 
        (u_p1 - u_p2) / dt,
    )

    # --- Implicit Dynamics ---
    f_impl = xdot - f_expl

    cont_dyn = vertcat(
        ode_flag * x3,
        ode_flag * x4,
        ode_flag * (-m * l * sin_theta * x4**2 + m * g * cos_theta * sin_theta + u) / denominator,
        ode_flag * (-m * l * cos_theta * sin_theta * x4**2 + u * cos_theta + (m_cart + m) * g * sin_theta) / (l * denominator),  
    )

    # --- Assemble Acados Model ---
    model = AcadosModel()
    model.name = model_name
    model.x = x_full
    model.xdot = xdot
    model.u = u
    model.p = p
    # model.f_expl = f_expl
    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl
    # model.disc_dyn_expr = x_next_full  # fallback Euler discretization

    ode = Function('ode', [x_curr, u, p], [cont_dyn])
    # set up RK4
    k1 = ode(x_curr,        u, p)
    k2 = ode(x_curr+dt/2*k1,u, p)
    k3 = ode(x_curr+dt/2*k2,u, p)
    k4 = ode(x_curr+dt*k3,  u, p)
    xf = x_curr + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    full_discrete = vertcat(
        xf,    # current state update (4 states), based on RK4 with ACTUAL control input (u)
        x1,    # previous state x1_p1 is just the current x1
        x2,     # previous state x2_p1 is just the current x2
        u,     # previous control input u_p1 is just the current u
        x1_p1, 
        x2_p1, 
        u_p1
    )

    model.disc_dyn_expr = full_discrete  # full discrete dynamics with past states

    # Optional labels
    model.x_labels = ['$x$', r'$\theta$', '$\dot{x}$', r'$\dot{\theta}$',
                      '$x_{-1}$', r'$\theta_{-1}$', '$\dot{x}_{-1}$', r'$\dot{\theta}_{-1}$',
                      '$x_{-2}$', r'$\theta_{-2}$', '$\dot{x}_{-2}$', r'$\dot{\theta}_{-2}$']
    model.u_labels = ['$F$']
    model.t_label = '$t$ [s]'

    return model

def export_ocp_cartpendulum(N, T, model, integrator_type, only_lower_bounds=False, **model_kwargs):
  
    # Limiti input
    lb_u = -50.0
    ub_u = 50.0

    Q = np.diagflat([15.0, 10.0, 0.9, 0.1])  # [cart, theta, cart_vel, omega]
    R = np.array([[0.5]])                    # [u]
    Qe = np.diagflat([10.0, 10.0, 0.1, 0.1])  # terminal cost

    # Definizione OCP
    ocp = AcadosOcp()
    ocp.model = model

    nx = model.x.shape[0]   # = 4
    nu = model.u.shape[0]   # = 1
    ny = 4 + nu            # output tracking
    ny_e = nx

    x0 = np.concatenate([np.array([0.0, np.pi]), np.zeros((nx-2))])

    ocp.dims.nx = nx
    ocp.dims.nu = nu
    ocp.dims.np = model.p.shape[0] if hasattr(model, "p") and isinstance(model.p, SX) else 0
    ocp.dims.ny = ny
    ocp.dims.ny_e = ny_e
    ocp.dims.ny_0 = ny

    #ocp.cost.cost_type = "NONLINEAR_LS"
    #ocp.cost.cost_type_e = "NONLINEAR_LS"
    #ocp.cost.cost_type_0 = "NONLINEAR_LS"
    #
    #ocp.model.cost_y_expr = vertcat(model.x[0:4], model.u)    
    #ocp.model.cost_y_expr_e = model.x[0:4]
    #
    #ocp.cost.W = block_diag(Q, R)
    #ocp.cost.W_0 = ocp.cost.W 
    #ocp.model.cost_y_expr_0 = vertcat(*[model.x[0], model.x[1], model.x[2], model.x[3], model.u])  
    #ocp.cost.yref_0 = np.zeros((int(ocp.model.cost_y_expr_0.shape[0]),))
    #ocp.cost.yref = np.zeros((5,))
    #ocp.cost.yref_e = np.zeros((4,))
    #
    #ocp.cost.W_e = Qe

    # cost
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.W = block_diag(Q, R)
    ocp.cost.W_0 = ocp.cost.W
    ocp.cost.W_e = Qe

    ocp.cost.Vx = np.eye(ny, nx)

    ocp.cost.Vu = np.zeros((ny, nu))
    ocp.cost.Vu[4,0] = np.eye(nu)

    ocp.cost.Vx_e = np.eye(ny_e)

    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))


    # vincoli
    ocp.constraints.constr_type = "BGH"
    ocp.constraints.lbu = np.array([lb_u])
    ocp.constraints.ubu = np.array([ub_u])
    ocp.constraints.idxbu = np.array(range(nu))
    ocp.constraints.x0 = x0

    # constraints on cart position
    ocp.constraints.idxbx = np.array([0])
    ocp.constraints.lbx = np.array([-10.0])
    ocp.constraints.ubx = np.array([10.0])

    ocp.dims.nh = 0

    # opzioni solver
    if integrator_type == "DISCRETE":
        ocp.solver_options.integrator_type = "DISCRETE"
    if integrator_type == "ERK":
        ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.tf = T
    ocp.solver_options.tol = 1e-2
    ocp.solver_options.N_horizon = N

    return ocp

def export_ocp_cartpendulum_actuation(N, T, model, only_lower_bounds=False, **model_kwargs):
    # initial state: cart at 0, pendulum upright
    x0 = np.array([0.0, np.pi, 0.0, 0.0, 0.0])

    # input bounds: force
    lb_u = -50.0
    ub_u = 50.0

    # cost weights
    Q = np.diagflat([10.0, 10.0, 0.1, 0.1])  # [cart, theta, cart_vel, omega]
    R = np.array([[0.3]])                    # [u]
    Qe = np.diagflat([10.0, 10.0, 0.1, 0.1])  # terminal cost

    dt = T / N

    # define OCP
    ocp = AcadosOcp()
    ocp.model = model
    ocp.model.p = model.p
    ocp.dims.N = N

    nx = model.x.shape[0]
    nu = model.u.shape[0]
    ny = 4 + nu
    ny_e = nx

    ocp.dims.nx = nx
    ocp.dims.nu = nu
    ocp.dims.np = model.p.shape[0] if hasattr(model, "p") and isinstance(model.p, SX) else 0
    ocp.dims.ny = ny
    ocp.dims.ny_e = ny_e
    ocp.dims.ny_0 = ny

    # ocp.cost.cost_type = "NONLINEAR_LS"
    # ocp.cost.cost_type_e = "NONLINEAR_LS"
    # ocp.cost.cost_type_0 = "NONLINEAR_LS"
    # 
    # ocp.model.cost_y_expr = vertcat(model.x[0:4], model.u)    
    # ocp.model.cost_y_expr_e = model.x[0:4]
    # 
    # ocp.cost.W = block_diag(Q, R)
    # ocp.cost.W_0 = ocp.cost.W 
    # ocp.model.cost_y_expr_0 = vertcat(*[model.x[0], model.x[1], model.x[2], model.x[3], model.u])  
    # ocp.cost.yref_0 = np.zeros((int(ocp.model.cost_y_expr_0.shape[0]),))
    # ocp.cost.yref = np.zeros((5,))
    # ocp.cost.yref_e = np.zeros((4,))
    # ocp.cost.W_e = Qe

    # cost
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    ocp.cost.W = block_diag(Q, R)
    ocp.cost.W_0 = ocp.cost.W
    ocp.cost.W_e = Qe

    # Updated for nx = 5 (one more state), still tracking only first 4 states and 1 control input
    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:4, :4] = np.eye(4)  # track only first 4 states

    ocp.cost.Vu = np.zeros((ny, nu))
    ocp.cost.Vu[4, 0] = 1.0  # control input

    ocp.cost.Vx_e = np.zeros((ny_e, nx))
    ocp.cost.Vx_e[:, :4] = np.eye(4)  # terminal cost on first 4 states

    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))
   
    # constraints
    ocp.constraints.constr_type = "BGH"
    ocp.constraints.lbu = np.array([lb_u])
    ocp.constraints.ubu = np.array([ub_u])
    ocp.constraints.idxbu = np.array(range(nu))
    ocp.constraints.x0 = x0

    # constraints on cart position
    ocp.constraints.idxbx = np.array([0])
    ocp.constraints.lbx = np.array([-10.0])
    ocp.constraints.ubx = np.array([10.0])

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

