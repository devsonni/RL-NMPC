import casadi as ca
from casadi import sin, cos, pi, tan
import numpy as np
from time import time
import matplotlib.pyplot as plt
from mayavi import mlab


# Shift function
def shift_timestep(T, t0, x0, u, rel_f_u, xs):
    # print(u[:, 0])
    f_value = rel_f_u(x0, u[:, 0])
    # print(f_value)
    x0 = ca.DM.full(x0 + (T * f_value))

    t0 = t0 + T
    u0 = ca.horzcat(
        u[:, 1:],
        ca.reshape(u[:, -1], -1, 1)
    )

    v = 12
    v_1 = 12
    v_2 = 12

    # con_t = [v, 0, v_1, 0, v_2, 0]  # Linear and angular velocity of target
    # if (mpc_iter >= 500):
    # con_t = [v, pi / 100, v_1, pi / 100, v_2, pi / 100]  # right turn
    # if (mpc_iter >= 500):
    #     con_t = [0, 5*pi / 4, 0, 5*pi / 4, 0, 5*pi / 4]  # 50 ahead
    # if (mpc_iter >= 501):
    #     con_t = [v, 0, v_1, 0, v_2, 0]  # left ahead
    # if (mpc_iter >= 906):
    #     con_t = [0, -5 * pi / 4, 0, -5 * pi / 4, 0, -5 * pi / 4]  # 50 ahead
    # if (mpc_iter >= 907):
    #     con_t = [v, -pi / 100, v_1, -pi / 100, v_2, -pi / 100]  # right turn
    # if (mpc_iter >= 907+500):
    #     con_t = [0, -5 * pi / 4, 0, -5 * pi / 4, 0, -5 * pi / 4]  # 50 ahead
    # if (mpc_iter >= 907+500+1):
    #     con_t = [v, 0, v_1, 0, v_2, 0]  # left ahead

    # con_t = [v, pi / 100, v_1, pi / 100, v_2, pi / 100]  # right turn
    # if (mpc_iter >= 541):
    #     con_t = [v, 0, v_1, 0, v_2, 0]  # left ahead
    # if (mpc_iter >= 895):
    #     con_t = [v, -pi / 100, v_1, -pi / 100, v_2, -pi / 100]  # 50 ahead
    # if (mpc_iter >= 1435):
    #     con_t = [v, 0, v_1, 0, v_2, 0]  # left ahead

    con_t = [v, 3 * pi / 200, v_1, 3 * pi / 200, v_2, 3 * pi / 200]  # right turn
    if (mpc_iter >= 500*2):
        con_t = [v, 0, v_1, 0, v_2, 0]  # left ahead
    if (mpc_iter >= (700 + 10 + 1 + 1)*2):
        con_t = [v, -3 * pi / 200, v_1, -3 * pi / 200, v_2, -3 * pi / 200]  # 50 ahead
    if (mpc_iter >= 2*(1200 + 10 + 1 + 1)):
        con_t = [v, 0, v_1, 0, v_2, 0]  # left ahead

    f_t_value = ca.vertcat(con_t[0] * cos(xs[2]),
                           con_t[0] * sin(xs[2]),
                           con_t[1],
                           con_t[2] * cos(xs[5]),
                           con_t[2] * sin(xs[5]),
                           con_t[3],
                           con_t[4] * cos(xs[8]),
                           con_t[4] * sin(xs[8]),
                           con_t[5],
                           )
    # print(xs)
    xs = ca.DM.full(xs + (T * f_t_value))
    # print(xs)
    return t0, x0, u0, xs


def DM2Arr(dm):
    return np.array(dm.full())


def SX2Arr(sx):
    return np.array(sx.full())


# For plotting a cylinder
def data_for_cylinder_along_z(center_x, center_y, radius, height_z):
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2 * np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + center_x
    y_grid = radius * np.sin(theta_grid) + center_y
    return x_grid, y_grid, z_grid


# For plotting ellipse
def ellipse(a_p, b_p, x_e_1, y_e_1):
    x = ca.DM.zeros((101))
    y = ca.DM.zeros((101))
    th = np.linspace(0, 2 * np.pi, 100)
    x = a_p * sin(th) + x_e_1
    y = b_p * cos(th) + y_e_1
    return x, y


# mpc parameters
T = 0.1  # discrete step
N = 5  # number of look ahead steps

# Constrains of UAV with gimbal
# input constrains of UAV
v_u_min = 0
v_u_max = 30
omega_2_u_min = - pi / 30
omega_2_u_max = pi / 30
omega_3_u_min = -1 * (pi / 21)
omega_3_u_max = 1 * (pi / 21)

v_u_min_1 = 14
v_u_max_1 = 30
omega_2_u_min_1 = - pi / 30
omega_2_u_max_1 = pi / 30
omega_3_u_min_1 = -7 * (pi / 21)
omega_3_u_max_1 = 7 * (pi / 21)

v_u_min_2 = 18
v_u_max_2 = 30
omega_2_u_min_2 = - pi / 30
omega_2_u_max_2 = pi / 30
omega_3_u_min_2 = -10.5 * (pi / 21)
omega_3_u_max_2 = 10.5 * (pi / 21)

# input constrains of gimbal
omega_1_g_min = -pi / 30
omega_1_g_max = pi / 30
omega_2_g_min = -pi / 30
omega_2_g_max = pi / 30
omega_3_g_min = -pi / 30
omega_3_g_max = pi / 30

omega_1_g_min_1 = -pi / 30
omega_1_g_max_1 = pi / 30
omega_2_g_min_1 = -pi / 30
omega_2_g_max_1 = pi / 30
omega_3_g_min_1 = -pi / 30
omega_3_g_max_1 = pi / 30

omega_1_g_min_2 = -pi / 30
omega_1_g_max_2 = pi / 30
omega_2_g_min_2 = -pi / 30
omega_2_g_max_2 = pi / 30
omega_3_g_min_2 = -pi / 30
omega_3_g_max_2 = pi / 30

# states constrains of UAV
theta_u_min = -0.2618
theta_u_max = 0.2618
z_u_min = 200
z_u_max = 200

theta_u_min_1 = -0.2618
theta_u_max_1 = 0.2618
z_u_min_1 = 200
z_u_max_1 = 200

theta_u_min_2 = -0.2618
theta_u_max_2 = 0.2618
z_u_min_2 = 200
z_u_max_2 = 200

# states constrains of gimbal
phi_g_min = -pi / 6
phi_g_max = pi / 6
theta_g_min = -pi / 6
theta_g_max = pi / 6
shi_g_min = -pi / 2
shi_g_max = pi / 2

phi_g_min_1 = -pi / 6
phi_g_max_1 = pi / 6
theta_g_min_1 = -pi / 6
theta_g_max_1 = pi / 6
shi_g_min_1 = -pi / 2
shi_g_max_1 = pi / 2

phi_g_min_2 = -pi / 6
phi_g_max_2 = pi / 6
theta_g_min_2 = -pi / 6
theta_g_max_2 = pi / 6
shi_g_min_2 = -pi / 2
shi_g_max_2 = pi / 2

# Symbolic states of UAV with gimbal camera

# states of the UAV
x_u = ca.SX.sym('x_u')
y_u = ca.SX.sym('y_u')
z_u = ca.SX.sym('z_u')
theta_u = ca.SX.sym('theta_u')
psi_u = ca.SX.sym('psi_u')
# states of the gimbal
phi_g = ca.SX.sym('phi_g')
shi_g = ca.SX.sym('shi_g')
theta_g = ca.SX.sym('theta_g')

# states of the UAV
x_u_1 = ca.SX.sym('x_u_1')
y_u_1 = ca.SX.sym('y_u_1')
z_u_1 = ca.SX.sym('z_u_1')
theta_u_1 = ca.SX.sym('theta_u_1')
psi_u_1 = ca.SX.sym('psi_u_1')
# states of the gimbal
phi_g_1 = ca.SX.sym('phi_g_1')
shi_g_1 = ca.SX.sym('shi_g_1')
theta_g_1 = ca.SX.sym('theta_g_1')

# states of the UAV
x_u_2 = ca.SX.sym('x_u_2')
y_u_2 = ca.SX.sym('y_u_2')
z_u_2 = ca.SX.sym('z_u_2')
theta_u_2 = ca.SX.sym('theta_u_2')
psi_u_2 = ca.SX.sym('psi_u_2')
# states of the gimbal
phi_g_2 = ca.SX.sym('phi_g_2')
shi_g_2 = ca.SX.sym('shi_g_2')
theta_g_2 = ca.SX.sym('theta_g_2')

# Real states
# states of the UAV
rel_x_u = ca.SX.sym('rel_x_u')
rel_y_u = ca.SX.sym('rel_y_u')
rel_z_u = ca.SX.sym('rel_z_u')
rel_theta_u = ca.SX.sym('rel_theta_u')
rel_psi_u = ca.SX.sym('rel_psi_u')
# states of the gimbal
rel_phi_g = ca.SX.sym('rel_phi_g')
rel_shi_g = ca.SX.sym('rel_shi_g')
rel_theta_g = ca.SX.sym('rel_theta_g')

# states of the UAV
rel_x_u_1 = ca.SX.sym('rel_x_u_1')
rel_y_u_1 = ca.SX.sym('rel_y_u_1')
rel_z_u_1 = ca.SX.sym('rel_z_u_1')
rel_theta_u_1 = ca.SX.sym('rel_theta_u_1')
rel_psi_u_1 = ca.SX.sym('rel_psi_u_1')
# states of the gimbal
rel_phi_g_1 = ca.SX.sym('rel_phi_g_1')
rel_shi_g_1 = ca.SX.sym('rel_shi_g_1')
rel_theta_g_1 = ca.SX.sym('rel_theta_g_1')

# states of the UAV
rel_x_u_2 = ca.SX.sym('rel_x_u_2')
rel_y_u_2 = ca.SX.sym('rel_y_u_2')
rel_z_u_2 = ca.SX.sym('rel_z_u_2')
rel_theta_u_2 = ca.SX.sym('rel_theta_u_2')
rel_psi_u_2 = ca.SX.sym('rel_psi_u_2')
# states of the gimbal
rel_phi_g_2 = ca.SX.sym('rel_phi_g_2')
rel_shi_g_2 = ca.SX.sym('rel_shi_g_2')
rel_theta_g_2 = ca.SX.sym('rel_theta_g_2')

rel_v_u_2 = ca.SX.sym('rel_v_u_2')
rel_v_u_1 = ca.SX.sym('rel_v_u_1')
rel_v_u = ca.SX.sym('rel_v_u')

a_u = ca.SX.sym('a_u')
a_u_1 = ca.SX.sym('a_u_1')
a_u_2 = ca.SX.sym('a_u_2')

# Controls of UAV that will find by NMPC
# UAV controls parameters
v_u = ca.SX.sym('v_u')
omega_2_u = ca.SX.sym('omega_2_u')
omega_3_u = ca.SX.sym('omega_3_u')
# Gimbal control parameters
omega_1_g = ca.SX.sym('omega_1_g')
omega_2_g = ca.SX.sym('omega_2_g')
omega_3_g = ca.SX.sym('omega_3_g')

# UAV controls parameters
v_u_1 = ca.SX.sym('v_u_1')
omega_2_u_1 = ca.SX.sym('omega_2_u_1')
omega_3_u_1 = ca.SX.sym('omega_3_u_1')
# Gimbal control parameters
omega_1_g_1 = ca.SX.sym('omega_1_g_1')
omega_2_g_1 = ca.SX.sym('omega_2_g_1')
omega_3_g_1 = ca.SX.sym('omega_3_g_1')

v_u_2 = ca.SX.sym('v_u_2')
omega_2_u_2 = ca.SX.sym('omega_2_u_2')
omega_3_u_2 = ca.SX.sym('omega_3_u_2')
# Gimbal control parameters
omega_1_g_2 = ca.SX.sym('omega_1_g_2')
omega_2_g_2 = ca.SX.sym('omega_2_g_2')
omega_3_g_2 = ca.SX.sym('omega_3_g_2')

# append all UAV states in one vector
states_u = ca.vertcat(
    x_u,
    y_u,
    z_u,
    theta_u,
    psi_u,
    phi_g,
    shi_g,
    theta_g,
    x_u_1,
    y_u_1,
    z_u_1,
    theta_u_1,
    psi_u_1,
    phi_g_1,
    shi_g_1,
    theta_g_1,
    x_u_2,
    y_u_2,
    z_u_2,
    theta_u_2,
    psi_u_2,
    phi_g_2,
    shi_g_2,
    theta_g_2,
    v_u,
    v_u_1,
    v_u_2
)

rel_states_u = ca.vertcat(
    rel_x_u,
    rel_y_u,
    rel_z_u,
    rel_theta_u,
    rel_psi_u,
    rel_phi_g,
    rel_shi_g,
    rel_theta_g,
    rel_x_u_1,
    rel_y_u_1,
    rel_z_u_1,
    rel_theta_u_1,
    rel_psi_u_1,
    rel_phi_g_1,
    rel_shi_g_1,
    rel_theta_g_1,
    rel_x_u_2,
    rel_y_u_2,
    rel_z_u_2,
    rel_theta_u_2,
    rel_psi_u_2,
    rel_phi_g_2,
    rel_shi_g_2,
    rel_theta_g_2,
    rel_v_u,
    rel_v_u_1,
    rel_v_u_2
)

n_states_u = states_u.numel()

# Appending controls in one vector
controls_u = ca.vertcat(
    a_u,
    omega_2_u,
    omega_3_u,
    omega_1_g,
    omega_2_g,
    omega_3_g,
    a_u_1,
    omega_2_u_1,
    omega_3_u_1,
    omega_1_g_1,
    omega_2_g_1,
    omega_3_g_1,
    a_u_2,
    omega_2_u_2,
    omega_3_u_2,
    omega_1_g_2,
    omega_2_g_2,
    omega_3_g_2
)
n_controls = controls_u.numel()

# Calculates RHS using control vector and current initial states of UAV and gimbal
rhs_u = ca.vertcat(
    v_u * cos(psi_u) * cos(theta_u),
    v_u * sin(psi_u) * cos(theta_u),
    v_u * sin(theta_u),
    omega_2_u,
    omega_3_u,
    omega_1_g,
    omega_2_g,
    omega_3_g,
    v_u_1 * cos(psi_u_1) * cos(theta_u_1),
    v_u_1 * sin(psi_u_1) * cos(theta_u_1),
    v_u_1 * sin(theta_u_1),
    omega_2_u_1,
    omega_3_u_1,
    omega_1_g_1,
    omega_2_g_1,
    omega_3_g_1,
    v_u_2 * cos(psi_u_2) * cos(theta_u_2),
    v_u_2 * sin(psi_u_2) * cos(theta_u_2),
    v_u_2 * sin(theta_u_2),
    omega_2_u_2,
    omega_3_u_2,
    omega_1_g_2,
    omega_2_g_2,
    omega_3_g_2,
    a_u,
    a_u_1,
    a_u_2
)

# real rhs

rel_rhs_u = ca.vertcat(
    rel_v_u * cos(rel_psi_u) * cos(rel_theta_u),
    0.7 + rel_v_u * sin(rel_psi_u) * cos(rel_theta_u),
    rel_v_u * sin(rel_theta_u),
    omega_2_u,
    omega_3_u,
    omega_1_g,
    omega_2_g,
    omega_3_g,
    rel_v_u_1 * cos(rel_psi_u_1) * cos(rel_theta_u_1),
    0.7 + rel_v_u_1 * sin(rel_psi_u_1) * cos(rel_theta_u_1),
    rel_v_u_1 * sin(rel_theta_u_1),
    omega_2_u_1,
    omega_3_u_1,
    omega_1_g_1,
    omega_2_g_1,
    omega_3_g_1,
    rel_v_u_2 * cos(rel_psi_u_2) * cos(rel_theta_u_2),
    0.7 + rel_v_u_2 * sin(rel_psi_u_2) * cos(rel_theta_u_2),
    rel_v_u_2 * sin(rel_theta_u_2),
    omega_2_u_2,
    omega_3_u_2,
    omega_1_g_2,
    omega_2_g_2,
    omega_3_g_2,
    a_u,
    a_u_1,
    a_u_2
)

# Non-linear mapping function which is f(x,y)
f_u = ca.Function('f', [states_u, controls_u], [rhs_u])
rel_f_u = ca.Function('rel_f_u', [rel_states_u, controls_u], [rel_rhs_u])

U = ca.SX.sym('U', n_controls, N)  # Decision Variables
P = ca.SX.sym('P', n_states_u + 9)  # This consists of initial states of UAV with gimbal 1-8 and
# reference states 9-11 (reference states is target's states)

X = ca.SX.sym('X', n_states_u, (N + 1))  # Has prediction of states over prediction horizon

# Filling the defined system parameters of UAV
X[:, 0] = P[0:27]  # initial state

for k in range(N):
    st = X[:, k]
    con = U[:, k]
    f_value = f_u(st, con)
    st_next = st + T * f_value
    X[:, k + 1] = st_next

ff = ca.Function('ff', [U, P], [X])

# Objective function

obj = 0  # objective function that need to minimize with optimal variables
g = []
# g = ca.SX.sym('g', 8*(N+1))  # constrains of pitch angle theta

count = 0
for k in range(N):
    stt = X[:, 0:N]
    A = ca.SX.sym('A', N)
    B = ca.SX.sym('B', N)
    C = ca.SX.sym('C', N)
    X_E = ca.SX.sym('X_E', N)
    Y_E = ca.SX.sym('Y_E', N)

    A_1 = ca.SX.sym('A_1', N)
    B_1 = ca.SX.sym('B_1', N)
    C_1 = ca.SX.sym('C_1', N)
    X_E_1 = ca.SX.sym('X_E_1', N)
    Y_E_1 = ca.SX.sym('Y_E_1', N)

    A_2 = ca.SX.sym('A_2', N)
    B_2 = ca.SX.sym('B_2', N)
    C_2 = ca.SX.sym('C_2', N)
    X_E_2 = ca.SX.sym('X_E_2', N)
    Y_E_2 = ca.SX.sym('Y_E_2', N)

    VFOV = 1  # Making FOV
    HFOV = 1
    VFOV_1 = 1  # Making FOV
    HFOV_1 = 1
    VFOV_2 = 1  # Making FOV
    HFOV_2 = 1

    w1 = 1  # MPC weights
    w2 = 1
    w3 = 1
    w4 = 1
    w5 = 1
    w6 = 1

    a = (stt[2, k] * (tan(stt[6, k] + VFOV / 2)) - stt[2, k] * tan(stt[6, k] - VFOV / 2)) / 2  # FOV Stuff
    b = (stt[2, k] * (tan(stt[5, k] + HFOV / 2)) - stt[2, k] * tan(stt[5, k] - HFOV / 2)) / 2

    A[k] = ((cos(stt[7, k])) ** 2) / a ** 2 + ((sin(stt[7, k])) ** 2) / b ** 2
    B[k] = 2 * cos(stt[7, k]) * sin(stt[7, k]) * ((1 / a ** 2) - (1 / b ** 2))
    C[k] = ((sin(stt[7, k])) ** 2) / a ** 2 + ((cos(stt[7, k])) ** 2) / b ** 2

    X_E[k] = stt[0, k] + a + stt[2, k] * (tan(stt[6, k] - VFOV / 2))  # Centre of FOV
    Y_E[k] = stt[1, k] + b + stt[2, k] * (tan(stt[5, k] - HFOV / 2))

    # 2
    a_1 = (stt[10, k] * (tan(stt[14, k] + VFOV_1 / 2)) - stt[10, k] * tan(stt[14, k] - VFOV_1 / 2)) / 2
    b_1 = (stt[10, k] * (tan(stt[13, k] + HFOV_1 / 2)) - stt[10, k] * tan(stt[13, k] - HFOV_1 / 2)) / 2

    A_1[k] = ((cos(stt[15, k])) ** 2) / a_1 ** 2 + ((sin(stt[15, k])) ** 2) / b_1 ** 2
    B_1[k] = 2 * cos(stt[15, k]) * sin(stt[15, k]) * ((1 / a_1 ** 2) - (1 / b_1 ** 2))
    C_1[k] = ((sin(stt[15, k])) ** 2) / a_1 ** 2 + ((cos(stt[15, k])) ** 2) / b_1 ** 2

    X_E_1[k] = stt[8, k] + a_1 + stt[10, k] * (tan(stt[14, k] - VFOV_1 / 2))  # Centre of FOV
    Y_E_1[k] = stt[9, k] + b_1 + stt[10, k] * (tan(stt[13, k] - HFOV_1 / 2))

    # 3
    a_2 = (stt[18, k] * (tan(stt[22, k] + VFOV_2 / 2)) - stt[18, k] * tan(stt[22, k] - VFOV_2 / 2)) / 2
    b_2 = (stt[18, k] * (tan(stt[21, k] + HFOV_2 / 2)) - stt[18, k] * tan(stt[21, k] - HFOV_2 / 2)) / 2

    A_2[k] = ((cos(stt[23, k])) ** 2) / a_2 ** 2 + ((sin(stt[23, k])) ** 2) / b_2 ** 2
    B_2[k] = 2 * cos(stt[23, k]) * sin(stt[23, k]) * ((1 / a_2 ** 2) - (1 / b_2 ** 2))
    C_2[k] = ((sin(stt[23, k])) ** 2) / a_2 ** 2 + ((cos(stt[23, k])) ** 2) / b_2 ** 2

    X_E_2[k] = stt[16, k] + a_2 + stt[18, k] * (tan(stt[22, k] - VFOV_2 / 2))  # Centre of FOV
    Y_E_2[k] = stt[17, k] + b_2 + stt[18, k] * (tan(stt[21, k] - HFOV_2 / 2))

    obj = obj + \
          w1 * ca.sqrt((stt[0, k] - P[27]) ** 2 + (stt[1, k] - P[28]) ** 2) + \
          w2 * ((A[k] * (P[27] - X_E[k]) ** 2 + B[k] * (P[28] - Y_E[k]) * (P[27] - X_E[k])
                 + C[k] * (P[28] - Y_E[k]) ** 2) - 1) + \
          w3 * ca.sqrt((stt[8, k] - P[30]) ** 2 + (stt[9, k] - P[31]) ** 2) + \
          w4 * ((A_1[k] * (P[30] - X_E_1[k]) ** 2 + B_1[k] * (P[31] - Y_E_1[k]) * (P[30] - X_E_1[k])
                 + C_1[k] * (P[31] - Y_E_1[k]) ** 2) - 1) + \
          w5 * ca.sqrt((stt[16, k] - P[33]) ** 2 + (stt[17, k] - P[34]) ** 2) + \
          w6 * ((A_2[k] * (P[33] - X_E_2[k]) ** 2 + B_2[k] * (P[34] - Y_E_2[k]) * (P[33] - X_E_2[k])
                 + C_2[k] * (P[34] - Y_E_2[k]) ** 2) - 1)

# Obstacle parameters and virtual radius of uav
x_o_1 = 90+100+60
y_o_1 = 400
x_o_2 = -75+20
y_o_2 = 500+20
x_o_3 = -425-20+15
y_o_3 = 400
x_o_4 = -160-100+30
y_o_4 = -10+50-30+20
x_o_5 = 15+100-30
y_o_5 = -10+50-30+20
x_o_6 = 200+70
y_o_6 = -330-30
x_o_8 = -390-20
y_o_8 = -330-30
x_o_7 = -80
y_o_7 = -480+20
x_o_9 = 1000
y_o_9 = 1000
x_o_10 = 1000
y_o_10 = 1000
obs_r = 30
UAV_r = 1

# compute the constrains, states or inequality constrains
for k in range(N + 1):
    g = ca.vertcat(g,
                   X[2, k],  # limit on height of UAV
                   X[3, k],  # limit on pitch angle theta
                   X[5, k],  # limit on gimbal angle phi
                   X[6, k],  # limit on gimbal angle theta
                   X[7, k],  # limit on gimbal angle shi
                   -ca.sqrt((X[0, k] - x_o_1) ** 2 + (X[1, k] - y_o_1) ** 2) + (UAV_r + obs_r),  # limit of obstacle-1
                   -ca.sqrt((X[0, k] - x_o_2) ** 2 + (X[1, k] - y_o_2) ** 2) + (UAV_r + obs_r),  # limit of obstacle-2
                   -ca.sqrt((X[0, k] - x_o_3) ** 2 + (X[1, k] - y_o_3) ** 2) + (UAV_r + obs_r),  # limit of obstacle-3
                   -ca.sqrt((X[0, k] - x_o_4) ** 2 + (X[1, k] - y_o_4) ** 2) + (UAV_r + obs_r),  # limit of obstacle-4
                   -ca.sqrt((X[0, k] - x_o_5) ** 2 + (X[1, k] - y_o_5) ** 2) + (UAV_r + obs_r),  # limit of obstacle-5
                   -ca.sqrt((X[0, k] - x_o_6) ** 2 + (X[1, k] - y_o_6) ** 2) + (UAV_r + obs_r),  # limit of obstacle-6
                   -ca.sqrt((X[0, k] - x_o_7) ** 2 + (X[1, k] - y_o_7) ** 2) + (UAV_r + obs_r),  # limit of obstacle-7
                   -ca.sqrt((X[0, k] - x_o_8) ** 2 + (X[1, k] - y_o_8) ** 2) + (UAV_r + obs_r),  # limit of obstacle-8
                   -ca.sqrt((X[0, k] - x_o_9) ** 2 + (X[1, k] - y_o_9) ** 2) + (UAV_r + obs_r),  # limit of obstacle-9
                   -ca.sqrt((X[0, k] - x_o_10) ** 2 + (X[1, k] - y_o_10) ** 2) + (UAV_r + obs_r), # limit of obstacle-10
                   X[2 + 8, k],  # limit on height of UAV
                   X[3 + 8, k],  # limit on pitch angle theta
                   X[5 + 8, k],  # limit on gimbal angle phi
                   X[6 + 8, k],  # limit on gimbal angle theta
                   X[7 + 8, k],  # limit on gimbal angle shi
                   -ca.sqrt((X[8, k] - x_o_1) ** 2 + (X[9, k] - y_o_1) ** 2) + (UAV_r + obs_r),  # limit of obstacle-1
                   -ca.sqrt((X[8, k] - x_o_2) ** 2 + (X[9, k] - y_o_2) ** 2) + (UAV_r + obs_r),  # limit of obstacle-2
                   -ca.sqrt((X[8, k] - x_o_3) ** 2 + (X[9, k] - y_o_3) ** 2) + (UAV_r + obs_r),  # limit of obstacle-3
                   -ca.sqrt((X[8, k] - x_o_4) ** 2 + (X[9, k] - y_o_4) ** 2) + (UAV_r + obs_r),  # limit of obstacle-4
                   -ca.sqrt((X[8, k] - x_o_5) ** 2 + (X[9, k] - y_o_5) ** 2) + (UAV_r + obs_r),  # limit of obstacle-5
                   -ca.sqrt((X[8, k] - x_o_6) ** 2 + (X[9, k] - y_o_6) ** 2) + (UAV_r + obs_r),  # limit of obstacle-6
                   -ca.sqrt((X[8, k] - x_o_7) ** 2 + (X[9, k] - y_o_7) ** 2) + (UAV_r + obs_r),  # limit of obstacle-7
                   -ca.sqrt((X[8, k] - x_o_8) ** 2 + (X[9, k] - y_o_8) ** 2) + (UAV_r + obs_r),  # limit of obstacle-8
                   -ca.sqrt((X[8, k] - x_o_9) ** 2 + (X[9, k] - y_o_9) ** 2) + (UAV_r + obs_r),  # limit of obstacle-9
                   -ca.sqrt((X[8, k] - x_o_10) ** 2 + (X[9, k] - y_o_10) ** 2) + (UAV_r + obs_r),
                   -ca.sqrt((X[8, k] - X[0, k]) ** 2 + (X[9, k] - X[1, k]) ** 2) + 2, # collusion with each other
                   X[2 + 8 + 8, k],  # limit on height of UAV
                   X[3 + 8 + 8, k],  # limit on pitch angle theta
                   X[5 + 8 + 8, k],  # limit on gimbal angle phi
                   X[6 + 8 + 8, k],  # limit on gimbal angle theta
                   X[7 + 8 + 8, k],  # limit on gimbal angle shi
                   -ca.sqrt((X[16, k] - x_o_1) ** 2 + (X[17, k] - y_o_1) ** 2) + (UAV_r + obs_r), # limit of obstacle-1
                   -ca.sqrt((X[16, k] - x_o_2) ** 2 + (X[17, k] - y_o_2) ** 2) + (UAV_r + obs_r), # limit of obstacle-2
                   -ca.sqrt((X[16, k] - x_o_3) ** 2 + (X[17, k] - y_o_3) ** 2) + (UAV_r + obs_r), # limit of obstacle-3
                   -ca.sqrt((X[16, k] - x_o_4) ** 2 + (X[17, k] - y_o_4) ** 2) + (UAV_r + obs_r), # limit of obstacle-4
                   -ca.sqrt((X[16, k] - x_o_5) ** 2 + (X[17, k] - y_o_5) ** 2) + (UAV_r + obs_r), # limit of obstacle-5
                   -ca.sqrt((X[16, k] - x_o_6) ** 2 + (X[17, k] - y_o_6) ** 2) + (UAV_r + obs_r), # limit of obstacle-6
                   -ca.sqrt((X[16, k] - x_o_7) ** 2 + (X[17, k] - y_o_7) ** 2) + (UAV_r + obs_r), # limit of obstacle-7
                   -ca.sqrt((X[16, k] - x_o_8) ** 2 + (X[17, k] - y_o_8) ** 2) + (UAV_r + obs_r), # limit of obstacle-8
                   -ca.sqrt((X[16, k] - x_o_9) ** 2 + (X[17, k] - y_o_9) ** 2) + (UAV_r + obs_r), # limit of obstacle-9
                   -ca.sqrt((X[16, k] - x_o_10) ** 2 + (X[17, k] - y_o_10) ** 2) + (UAV_r + obs_r),
                   -ca.sqrt((X[16, k] - X[0, k]) ** 2 + (X[17, k] - X[1, k]) ** 2) + 2,
                   -ca.sqrt((X[16, k] - X[8, k]) ** 2 + (X[17, k] - X[9, k]) ** 2) + 2,
                   X[24, k],
                   X[25, k],
                   X[26, k], #51
                   # ((ca.fabs((((X[9, k]-P[31])/(X[8, k]-P[30]))*(x_o_1-P[30]))+P[31]-y_o_1))/ca.sqrt(
                   #     ((X[9, k]-P[31])/(X[8, k]-P[30]))**2+1))-obs_r,
                   # ((ca.fabs((((X[9, k]-P[31])/(X[8, k]-P[30]))*(x_o_2-P[30]))+P[31]-y_o_2))/ca.sqrt(
                   #     ((X[9, k]-P[31])/(X[8, k]-P[30]))**2+1))-obs_r,
                   # ((ca.fabs((((X[9, k]-P[31])/(X[8, k]-P[30]))*(x_o_3-P[30]))+P[31]-y_o_3))/ca.sqrt(
                   #     ((X[9, k]-P[31])/(X[8, k]-P[30]))**2+1))-obs_r,
                   # ((ca.fabs((((X[9, k]-P[31])/(X[8, k]-P[30]))*(x_o_4-P[30]))+P[31]-y_o_4))/ca.sqrt(
                   #     ((X[9, k]-P[31])/(X[8, k]-P[30]))**2+1))-obs_r,
                   # ((ca.fabs((((X[9, k] - P[31]) / (X[8, k] - P[30])) * (x_o_5 - P[30])) + P[31] - y_o_5)) / ca.sqrt(
                   #     ((X[9, k] - P[31]) / (X[8, k] - P[30])) ** 2 + 1)) - obs_r,
                   # ((ca.fabs((((X[9, k] - P[31]) / (X[8, k] - P[30])) * (x_o_6 - P[30])) + P[31] - y_o_6)) / ca.sqrt(
                   #     ((X[9, k] - P[31]) / (X[8, k] - P[30])) ** 2 + 1)) - obs_r,
                   # ((ca.fabs((((X[9, k] - P[31]) / (X[8, k] - P[30])) * (x_o_7 - P[30])) + P[31] - y_o_7)) / ca.sqrt(
                   #     ((X[9, k] - P[31]) / (X[8, k] - P[30])) ** 2 + 1)) - obs_r,
                   # ((ca.fabs((((X[9, k] - P[31]) / (X[8, k] - P[30])) * (x_o_8 - P[30])) + P[31] - y_o_8)) / ca.sqrt(
                   #     ((X[9, k] - P[31]) / (X[8, k] - P[30])) ** 2 + 1)) - obs_r,
                   )

# 48
# make the decision variables one column vector
OPT_variables = \
    U.reshape((-1, 1))  # Example: 6x15 ---> 90x1 where 6=controls, 16=n+1

nlp_prob = {
    'f': obj,
    'x': OPT_variables,
    'g': g,
    'p': P
}

opts = {
    'ipopt': {
        'max_iter': 7,
        'print_level': 0,
        'acceptable_tol': 1e-8,
        'acceptable_obj_change_tol': 1e-6
    },
    'print_time': 0
}

solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

K = 51
lbx = ca.DM.zeros((n_controls * N, 1))
ubx = ca.DM.zeros((n_controls * N, 1))
lbg = ca.DM.zeros((K * (N + 1)))
ubg = ca.DM.zeros((K * (N + 1)))

# Constrains on states (Inequality constrains)
lbg[0:K * (N + 1):K] = z_u_min  # z lower bound
lbg[1:K * (N + 1):K] = theta_u_min  # theta lower bound
lbg[2:K * (N + 1):K] = phi_g_min  # phi lower bound
lbg[3:K * (N + 1):K] = theta_g_min  # theta lower bound
lbg[4:K * (N + 1):K] = shi_g_min  # shi lower bound
lbg[6:K * (N + 1):K] = -ca.inf  # Obstacle - 1
lbg[5:K * (N + 1):K] = -ca.inf  # Obstacle - 2
lbg[7:K * (N + 1):K] = -ca.inf  # Obstacle - 3
lbg[8:K * (N + 1):K] = -ca.inf  # Obstacle - 4
lbg[9:K * (N + 1):K] = -ca.inf  # Obstacle - 5
lbg[10:K * (N + 1):K] = -ca.inf  # Obstacle - 6
lbg[11:K * (N + 1):K] = -ca.inf  # Obstacle - 7
lbg[12:K * (N + 1):K] = -ca.inf  # Obstacle - 8
lbg[13:K * (N + 1):K] = -ca.inf  # Obstacle - 9
lbg[14:K * (N + 1):K] = -ca.inf  # Obstacle - 10
lbg[15:K * (N + 1):K] = z_u_min_1  # z lower bound
lbg[16:K * (N + 1):K] = theta_u_min_1  # theta lower bound
lbg[17:K * (N + 1):K] = phi_g_min_1  # phi lower bound
lbg[18:K * (N + 1):K] = theta_g_min_1  # theta lower bound
lbg[19:K * (N + 1):K] = shi_g_min_1  # shi lower bound
lbg[20:K * (N + 1):K] = -ca.inf  # Obstacle - 1
lbg[21:K * (N + 1):K] = -ca.inf  # Obstacle - 2
lbg[22:K * (N + 1):K] = -ca.inf  # Obstacle - 3
lbg[23:K * (N + 1):K] = -ca.inf  # Obstacle - 4
lbg[24:K * (N + 1):K] = -ca.inf  # Obstacle - 5
lbg[25:K * (N + 1):K] = -ca.inf  # Obstacle - 6
lbg[26:K * (N + 1):K] = -ca.inf  # Obstacle - 7
lbg[27:K * (N + 1):K] = -ca.inf  # Obstacle - 8
lbg[28:K * (N + 1):K] = -ca.inf  # Obstacle - 9
lbg[29:K * (N + 1):K] = -ca.inf  # Obstacle - 10
lbg[30:K * (N + 1):K] = -ca.inf  # inter collusion
lbg[31:K * (N + 1):K] = z_u_min_2  # z lower bound
lbg[32:K * (N + 1):K] = theta_u_min_2  # theta lower bound
lbg[33:K * (N + 1):K] = phi_g_min_2  # phi lower bound
lbg[34:K * (N + 1):K] = theta_g_min_2  # theta lower bound
lbg[35:K * (N + 1):K] = shi_g_min_2  # shi lower bound
lbg[36:K * (N + 1):K] = -ca.inf  # Obstacle - 1
lbg[37:K * (N + 1):K] = -ca.inf  # Obstacle - 2
lbg[38:K * (N + 1):K] = -ca.inf  # Obstacle - 3
lbg[39:K * (N + 1):K] = -ca.inf  # Obstacle - 4
lbg[40:K * (N + 1):K] = -ca.inf  # Obstacle - 5
lbg[41:K * (N + 1):K] = -ca.inf  # Obstacle - 6
lbg[42:K * (N + 1):K] = -ca.inf  # Obstacle - 7
lbg[43:K * (N + 1):K] = -ca.inf  # Obstacle - 8
lbg[44:K * (N + 1):K] = -ca.inf  # Obstacle - 9
lbg[45:K * (N + 1):K] = -ca.inf  # Obstacle - 10
lbg[46:K * (N + 1):K] = -ca.inf  # inter collusion
lbg[47:K * (N + 1):K] = -ca.inf  # inter collusion
lbg[48:K * (N + 1):K] = v_u_min
lbg[49:K * (N + 1):K] = v_u_min_1
lbg[50:K * (N + 1):K] = v_u_min_2
# lbg[51:K * (N + 1):K] = 0
# lbg[52:K * (N + 1):K] = 0
# lbg[53:K * (N + 1):K] = 0
# lbg[54:K * (N + 1):K] = 0
# lbg[55:K * (N + 1):K] = 0
# lbg[56:K * (N + 1):K] = 0
# lbg[57:K * (N + 1):K] = 0
# lbg[58:K * (N + 1):K] = 0

ubg[0:K * (N + 1):K] = z_u_max  # z lower bound
ubg[1:K * (N + 1):K] = theta_u_max  # theta lower bound
ubg[2:K * (N + 1):K] = phi_g_max  # phi lower bound
ubg[3:K * (N + 1):K] = theta_g_max  # theta lower bound
ubg[4:K * (N + 1):K] = shi_g_max  # shi lower bound
ubg[6:K * (N + 1):K] = 0  # Obstacle - 1
ubg[5:K * (N + 1):K] = 0  # Obstacle - 2
ubg[7:K * (N + 1):K] = 0  # Obstacle - 3
ubg[8:K * (N + 1):K] = 0  # Obstacle - 4
ubg[9:K * (N + 1):K] = 0  # Obstacle - 5
ubg[10:K * (N + 1):K] = 0  # Obstacle - 6
ubg[11:K * (N + 1):K] = 0  # Obstacle - 7
ubg[12:K * (N + 1):K] = 0  # Obstacle - 8
ubg[13:K * (N + 1):K] = 0  # Obstacle - 9
ubg[14:K * (N + 1):K] = 0  # Obstacle - 10
ubg[15:K * (N + 1):K] = z_u_max_1  # z lower bound
ubg[16:K * (N + 1):K] = theta_u_max_1  # theta lower bound
ubg[17:K * (N + 1):K] = phi_g_max_1  # phi lower bound
ubg[18:K * (N + 1):K] = theta_g_max_1  # theta lower bound
ubg[19:K * (N + 1):K] = shi_g_max_1  # shi lower bound
ubg[20:K * (N + 1):K] = 0  # Obstacle - 1
ubg[21:K * (N + 1):K] = 0  # Obstacle - 2
ubg[22:K * (N + 1):K] = 0  # Obstacle - 3
ubg[23:K * (N + 1):K] = 0  # Obstacle - 4
ubg[24:K * (N + 1):K] = 0  # Obstacle - 5
ubg[25:K * (N + 1):K] = 0  # Obstacle - 6
ubg[26:K * (N + 1):K] = 0  # Obstacle - 7
ubg[27:K * (N + 1):K] = 0  # Obstacle - 8
ubg[28:K * (N + 1):K] = 0  # Obstacle - 9
ubg[29:K * (N + 1):K] = 0  # Obstacle - 10
ubg[30:K * (N + 1):K] = 0  # inter collusion
ubg[31:K * (N + 1):K] = z_u_max_2  # z lower bound
ubg[32:K * (N + 1):K] = theta_u_max_2  # theta lower bound
ubg[33:K * (N + 1):K] = phi_g_max_2  # phi lower bound
ubg[34:K * (N + 1):K] = theta_g_max_2  # theta lower bound
ubg[35:K * (N + 1):K] = shi_g_max_2  # shi lower bound
ubg[36:K * (N + 1):K] = 0  # Obstacle - 1
ubg[37:K * (N + 1):K] = 0  # Obstacle - 2
ubg[38:K * (N + 1):K] = 0  # Obstacle - 3
ubg[39:K * (N + 1):K] = 0  # Obstacle - 4
ubg[40:K * (N + 1):K] = 0  # Obstacle - 5
ubg[41:K * (N + 1):K] = 0  # Obstacle - 6
ubg[42:K * (N + 1):K] = 0  # Obstacle - 7
ubg[43:K * (N + 1):K] = 0  # Obstacle - 8
ubg[44:K * (N + 1):K] = 0  # Obstacle - 9
ubg[45:K * (N + 1):K] = 0  # Obstacle - 10
ubg[46:K * (N + 1):K] = 0  # inter collusion
ubg[47:K * (N + 1):K] = 0  # inter collusion
ubg[48:K * (N + 1):K] = v_u_max
ubg[49:K * (N + 1):K] = v_u_max_1
ubg[50:K * (N + 1):K] = v_u_max_2
# ubg[51:K * (N + 1):K] = ca.inf
# ubg[52:K * (N + 1):K] = ca.inf
# ubg[53:K * (N + 1):K] = ca.inf
# ubg[54:K * (N + 1):K] = ca.inf
# ubg[55:K * (N + 1):K] = ca.inf
# ubg[56:K * (N + 1):K] = ca.inf
# ubg[57:K * (N + 1):K] = ca.inf
# ubg[58:K * (N + 1):K] = ca.inf

# Constrains on controls, constrains on optimization variable
lbx[0: n_controls * N: n_controls, 0] = -10  # acceleration lower bound
lbx[1: n_controls * N: n_controls, 0] = omega_2_u_min  # theta 1 lower bound
lbx[2: n_controls * N: n_controls, 0] = omega_3_u_min  # theta 2 lower bound
lbx[3: n_controls * N: n_controls, 0] = omega_1_g_min  # omega 1 lower bound
lbx[4: n_controls * N: n_controls, 0] = omega_2_g_min  # omega 2 lower bound
lbx[5: n_controls * N: n_controls, 0] = omega_3_g_min  # omega 3 lower bound
lbx[6: n_controls * N: n_controls, 0] = -10  # acceleration lower bound
lbx[7: n_controls * N: n_controls, 0] = omega_2_u_min_1  # theta 1 lower bound
lbx[8: n_controls * N: n_controls, 0] = omega_3_u_min_1  # theta 2 lower bound
lbx[9: n_controls * N: n_controls, 0] = omega_1_g_min_1  # omega 1 lower bound
lbx[10: n_controls * N: n_controls, 0] = omega_2_g_min_1  # omega 2 lower bound
lbx[11: n_controls * N: n_controls, 0] = omega_3_g_min_1  # omega 3 lower bound
lbx[12: n_controls * N: n_controls, 0] = -10  # acceleration lower bound
lbx[13: n_controls * N: n_controls, 0] = omega_2_u_min_2  # theta 1 lower bound
lbx[14: n_controls * N: n_controls, 0] = omega_3_u_min_2  # theta 2 lower bound
lbx[15: n_controls * N: n_controls, 0] = omega_1_g_min_2  # omega 1 lower bound
lbx[16: n_controls * N: n_controls, 0] = omega_2_g_min_2  # omega 2 lower bound
lbx[17: n_controls * N: n_controls, 0] = omega_3_g_min_2  # omega 3 lower bound

ubx[0: n_controls * N: n_controls, 0] = 10  # acceleration upper bound
ubx[1: n_controls * N: n_controls, 0] = omega_2_u_max  # theta 1 upper bound
ubx[2: n_controls * N: n_controls, 0] = omega_3_u_max  # theta 2 upper bound
ubx[3: n_controls * N: n_controls, 0] = omega_1_g_max  # omega 1 upper bound
ubx[4: n_controls * N: n_controls, 0] = omega_2_g_max  # omega 2 upper bound
ubx[5: n_controls * N: n_controls, 0] = omega_3_g_max  # omega 3 upper bound
ubx[6: n_controls * N: n_controls, 0] = 10  # acceleration upper bound
ubx[7: n_controls * N: n_controls, 0] = omega_2_u_max_1  # theta 1 upper bound
ubx[8: n_controls * N: n_controls, 0] = omega_3_u_max_1  # theta 2 upper bound
ubx[9: n_controls * N: n_controls, 0] = omega_1_g_max_1  # omega 1 upper bound
ubx[10: n_controls * N: n_controls, 0] = omega_2_g_max_1  # omega 2 upper bound
ubx[11: n_controls * N: n_controls, 0] = omega_3_g_max_1  # omega 3 upper bound
ubx[12: n_controls * N: n_controls, 0] = 10  # acceleration upper bound
ubx[13: n_controls * N: n_controls, 0] = omega_2_u_max_2  # theta 1 upper bound
ubx[14: n_controls * N: n_controls, 0] = omega_3_u_max_2  # theta 2 upper bound
ubx[15: n_controls * N: n_controls, 0] = omega_1_g_max_2  # omega 1 upper bound
ubx[16: n_controls * N: n_controls, 0] = omega_2_g_max_2  # omega 2 upper bound
ubx[17: n_controls * N: n_controls, 0] = omega_3_g_max_2  # omega 3 upper bound

args = {
    'lbg': lbg,  # lower bound for state
    'ubg': ubg,  # upper bound for state
    'lbx': lbx,  # lower bound for controls
    'ubx': ubx  # upper bound for controls
}

# target
x_target = 100
y_target = 150
theta_target = pi/4
x_target_1 = 100
y_target_1 = 200
theta_target_1 = pi/4
x_target_2 = 100
y_target_2 = 250
theta_target_2 = pi/4

t0 = 0
x0 = [99, 150, 200, 0, 0, 0, 0, 0, 99, 200, 200, 0, 0, 0, 0, 0, 99, 250, 200, 0, 0, 0, 0, 0, 0, 0, 0]
xs = [100, 150, pi/4, 100, 200, pi/4, 100, 250, pi/4]

t = ca.DM(t0)
loop_run = 1426*2

u0 = ca.DM.zeros((n_controls, N))  # initial control
xx = ca.DM.zeros((27, loop_run + 1))  # change size according to main loop run, works as tracker for target and UAV
ss = ca.DM.zeros((9, loop_run + 1))
x_e_1 = ca.DM.zeros((loop_run + 1))
y_e_1 = ca.DM.zeros((loop_run + 1))
x_e_2 = ca.DM.zeros((loop_run + 1))
y_e_2 = ca.DM.zeros((loop_run + 1))
x_e_5 = ca.DM.zeros((loop_run + 1))
y_e_5 = ca.DM.zeros((loop_run + 1))
x_e_6 = ca.DM.zeros((loop_run + 1))
y_e_6 = ca.DM.zeros((loop_run + 1))
UAV1_FOV_Plot_WRL = ca.DM.zeros((loop_run + 1, 4))
UAV2_FOV_Plot_WRL = ca.DM.zeros((loop_run + 1, 4))
UAV3_FOV_Plot_WRL = ca.DM.zeros((loop_run + 1, 4))
controls_uu = ca.DM.zeros((18, loop_run))
##
max_step_size = loop_run

linear_vel_min = np.zeros((max_step_size))
linear_vel_max = np.zeros((max_step_size))

pitch_rate_min = np.zeros((max_step_size))
pitch_rate_max = np.zeros((max_step_size))

yaw_rate_min = np.zeros((max_step_size))
yaw_rate_max = np.zeros((max_step_size))

groll_rate_min = np.zeros((max_step_size))
groll_rate_max = np.zeros((max_step_size))

gpitch_rate_min = np.zeros((max_step_size))
gpitch_rate_max = np.zeros((max_step_size))

gyaw_rate_min = np.zeros((max_step_size))
gyaw_rate_max = np.zeros((max_step_size))

z_aav_min = np.zeros((max_step_size))
z_aav_max = np.zeros((max_step_size))

##

linear_vel_min_1 = np.zeros((max_step_size))
linear_vel_max_1 = np.zeros((max_step_size))

linear_acc_min = np.zeros((max_step_size))
linear_acc_max = np.zeros((max_step_size))

pitch_rate_min_1 = np.zeros((max_step_size))
pitch_rate_max_1 = np.zeros((max_step_size))

yaw_rate_min_1 = np.zeros((max_step_size))
yaw_rate_max_1 = np.zeros((max_step_size))

groll_rate_min_1 = np.zeros((max_step_size))
groll_rate_max_1 = np.zeros((max_step_size))

gpitch_rate_min_1 = np.zeros((max_step_size))
gpitch_rate_max_1 = np.zeros((max_step_size))

gyaw_rate_min_1 = np.zeros((max_step_size))
gyaw_rate_max_1 = np.zeros((max_step_size))

z_aav_min_1 = np.zeros((max_step_size))
z_aav_max_1 = np.zeros((max_step_size))

##

linear_vel_min_2 = np.zeros((max_step_size))
linear_vel_max_2 = np.zeros((max_step_size))

pitch_rate_min_2 = np.zeros((max_step_size))
pitch_rate_max_2 = np.zeros((max_step_size))

yaw_rate_min_2 = np.zeros((max_step_size))
yaw_rate_max_2 = np.zeros((max_step_size))

groll_rate_min_2 = np.zeros((max_step_size))
groll_rate_max_2 = np.zeros((max_step_size))

gpitch_rate_min_2 = np.zeros((max_step_size))
gpitch_rate_max_2 = np.zeros((max_step_size))

gyaw_rate_min_2 = np.zeros((max_step_size))
gyaw_rate_max_2 = np.zeros((max_step_size))

z_aav_min_2 = np.zeros((max_step_size))
z_aav_max_2 = np.zeros((max_step_size))


xx[:, 0] = x0
ss[:, 0] = xs

times = ca.DM.zeros((loop_run))
for i in range(loop_run):
    times[i] = i * 0.2

times = np.array(times)
times1 = np.squeeze(times)

mpc_iter = 0
cat_controls = DM2Arr(u0[:, 0])
times = np.array([[0]])

for i in range(max_step_size):
    linear_vel_min[i] = 0
    linear_vel_max[i] = 30
    linear_acc_min[i] = -10
    linear_acc_max[i] = 10
    pitch_rate_min[i] = -pi / 30
    pitch_rate_max[i] = pi / 30
    yaw_rate_min[i] = -pi / 21
    yaw_rate_max[i] = pi / 21
    groll_rate_min[i] = -pi / 30
    groll_rate_max[i] = pi / 30
    gpitch_rate_min[i] = -pi / 30
    gpitch_rate_max[i] = pi / 30
    gyaw_rate_min[i] = -pi / 30
    gyaw_rate_max[i] = pi / 30
    z_aav_min[i] = 200
    z_aav_max[i] = 200

    linear_vel_min_1[i] = 14
    linear_vel_max_1[i] = 30
    linear_acc_min[i] = -10
    linear_acc_max[i] = 10
    pitch_rate_min_1[i] = -pi / 30
    pitch_rate_max_1[i] = pi / 30
    yaw_rate_min_1[i] = -7 * (pi / 21)
    yaw_rate_max_1[i] = 7 * (pi / 21)
    groll_rate_min_1[i] = -pi / 30
    groll_rate_max_1[i] = pi / 30
    gpitch_rate_min_1[i] = -pi / 30
    gpitch_rate_max_1[i] = pi / 30
    gyaw_rate_min_1[i] = -pi / 30
    gyaw_rate_max_1[i] = pi / 30
    z_aav_min_1[i] = 200
    z_aav_max_1[i] = 200

    linear_vel_min_2[i] = 18
    linear_vel_max_2[i] = 30
    linear_acc_min[i] = -10
    linear_acc_max[i] = 10
    pitch_rate_min_2[i] = -pi / 30
    pitch_rate_max_2[i] = pi / 30
    yaw_rate_min_2[i] = -10.5 * (pi / 21)
    yaw_rate_max_2[i] = 10.5 * (pi / 21)
    groll_rate_min_2[i] = -pi / 30
    groll_rate_max_2[i] = pi / 30
    gpitch_rate_min_2[i] = -pi / 30
    gpitch_rate_max_2[i] = pi / 30
    gyaw_rate_min_2[i] = -pi / 30
    gyaw_rate_max_2[i] = pi / 30
    z_aav_min_2[i] = 200
    z_aav_max_2[i] = 200



###############################################################################
##############################   Main Loop    #################################

if __name__ == '__main__':
    main_loop = time()  # return time in sec
    while mpc_iter < loop_run:
        # print('xs: {}'.format(xs))
        t1 = time()
        args['p'] = ca.vertcat(
            x0,  # current state
            xs  # target state
        )
        # print(args['p'])
        # optimization variable current state
        args['x0'] = \
            ca.reshape(u0, n_controls * N, 1)

        sol = solver(
            x0=args['x0'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p=args['p']
        )

        u = ca.reshape(sol['x'], n_controls, N)
        ff_value = ff(u, args['p'])

        cat_controls = np.vstack((
            cat_controls,
            DM2Arr(u[:, 0])
        ))

        controls_uu[:, mpc_iter] = u[:, 0]
        # print(cat_controls)

        t = np.vstack((
            t,
            t0
        ))
        # print(u[:,0])

        t0, x0, u0, xs = shift_timestep(T, t0, x0, u, rel_f_u, xs)

        # tracking states of target and UAV for plotting
        xx[:, mpc_iter + 1] = x0
        ss[:, mpc_iter + 1] = xs

        # xx ...
        t2 = time()
        # print(mpc_iter)
        # print(t2-t1)
        times = np.vstack((
            times,
            t2 - t1
        ))

        mpc_iter = mpc_iter + 1
        print(mpc_iter)
        # print(x0)

        a_p = (x0[2] * (tan(x0[6] + VFOV / 2)) - x0[2] * tan(x0[6] - VFOV / 2)) / 2  # For plotting FOV
        b_p = (x0[2] * (tan(x0[5] + HFOV / 2)) - x0[2] * tan(x0[5] - HFOV / 2)) / 2
        x_e_1[mpc_iter] = x0[0] + a_p + x0[2] * (tan(x0[6] - VFOV / 2))
        y_e_1[mpc_iter] = x0[1] + b_p + x0[2] * (tan(x0[5] - HFOV / 2))

        a_p_1 = (x0[10] * (tan(x0[14] + VFOV_1 / 2)) - x0[10] * tan(x0[14] - VFOV_1 / 2)) / 2
        b_p_1 = (x0[10] * (tan(x0[13] + HFOV_1 / 2)) - x0[10] * tan(x0[13] - HFOV_1 / 2)) / 2
        x_e_5[mpc_iter] = x0[8] + a_p_1 + x0[10] * (tan(x0[14] - VFOV_1 / 2))
        y_e_5[mpc_iter] = x0[9] + b_p_1 + x0[10] * (tan(x0[13] - HFOV_1 / 2))

        a_p_2 = (x0[18] * (tan(x0[22] + VFOV_2 / 2)) - x0[18] * tan(x0[22] - VFOV_2 / 2)) / 2
        b_p_2 = (x0[18] * (tan(x0[21] + HFOV_2 / 2)) - x0[18] * tan(x0[21] - HFOV_2 / 2)) / 2
        x_e_6[mpc_iter] = x0[16] + a_p_2 + x0[18] * (tan(x0[22] - VFOV_2 / 2))
        y_e_6[mpc_iter] = x0[17] + b_p_2 + x0[18] * (tan(x0[21] - HFOV_2 / 2))

        UAV1_FOV_Plot_WRL[mpc_iter, 0] = a_p
        UAV1_FOV_Plot_WRL[mpc_iter, 1] = b_p
        UAV1_FOV_Plot_WRL[mpc_iter, 2] = x_e_1[mpc_iter]
        UAV1_FOV_Plot_WRL[mpc_iter, 3] = y_e_1[mpc_iter]

        UAV2_FOV_Plot_WRL[mpc_iter, 0] = a_p_1
        UAV2_FOV_Plot_WRL[mpc_iter, 1] = b_p_1
        UAV2_FOV_Plot_WRL[mpc_iter, 2] = x_e_5[mpc_iter]
        UAV2_FOV_Plot_WRL[mpc_iter, 3] = y_e_5[mpc_iter]

        UAV3_FOV_Plot_WRL[mpc_iter, 0] = a_p_2
        UAV3_FOV_Plot_WRL[mpc_iter, 1] = b_p_2
        UAV3_FOV_Plot_WRL[mpc_iter, 2] = x_e_6[mpc_iter]
        UAV3_FOV_Plot_WRL[mpc_iter, 3] = y_e_6[mpc_iter]

        # print('fovx: {}'.format(x_e_1[mpc_iter]))
        # print('fovy: {}'.format(y_e_1[mpc_iter]))

################################################################################
################################# Plotting Stuff ###############################

# Collecting data for obstacles
Xc_1, Yc_1, Zc_1 = data_for_cylinder_along_z(x_o_1, y_o_1, obs_r, 250)
Xc_2, Yc_2, Zc_2 = data_for_cylinder_along_z(x_o_2, y_o_2, obs_r, 230)
Xc_3, Yc_3, Zc_3 = data_for_cylinder_along_z(x_o_3, y_o_3, obs_r, 250)
Xc_4, Yc_4, Zc_4 = data_for_cylinder_along_z(x_o_4, y_o_4, obs_r, 230)
Xc_5, Yc_5, Zc_5 = data_for_cylinder_along_z(x_o_5, y_o_5, obs_r, 250)
Xc_6, Yc_6, Zc_6 = data_for_cylinder_along_z(x_o_6, y_o_6, obs_r, 230)
Xc_7, Yc_7, Zc_7 = data_for_cylinder_along_z(x_o_7, y_o_7, obs_r, 250)
Xc_8, Yc_8, Zc_8 = data_for_cylinder_along_z(x_o_8, y_o_8, obs_r, 230)
Xc_9, Yc_9, Zc_9 = data_for_cylinder_along_z(x_o_9, y_o_9, obs_r, 250)
Xc_10, Yc_10, Zc_10 = data_for_cylinder_along_z(x_o_10, y_o_10, obs_r, 230)
x_e, y_e = ellipse(a_p, b_p, x_e_1[mpc_iter], y_e_1[mpc_iter])
x_e_2nd, y_e_2nd = ellipse(a_p_1, b_p_1, x_e_5[mpc_iter], y_e_5[mpc_iter])
x_e_3nd, y_e_3nd = ellipse(a_p_2, b_p_2, x_e_6[mpc_iter], y_e_6[mpc_iter])

# Plotting starts from here
xx1 = np.array(xx)
xs1 = np.array(ss)
ss1 = np.zeros((loop_run + 1))
x_e_2 = np.array(x_e)
y_e_2 = np.array(y_e)
x_e_3 = np.array(x_e_2nd)
y_e_3 = np.array(y_e_2nd)
x_e_7 = np.array(x_e_3nd)
y_e_7 = np.array(y_e_3nd)
controls_uu = np.array(controls_uu)

ss2 = np.zeros((loop_run + 1, 1))

# print(x_e_1[0:loop_run - 1].shape)
# print(y_e_1[0:loop_run - 1].shape)
# print(ss1[0:loop_run - 1].shape)

fig = plt.figure(figsize=(14, 9))
ax = plt.axes(projection='3d')
ax.plot3D(xx1[0, 0:loop_run - 1], xx1[1, 0:loop_run - 1], xx1[2, 0:loop_run - 1], linewidth="3", color="red")
ax.plot3D(xs1[0, 0:loop_run - 1], xs1[1, 0:loop_run - 1], ss1[0:loop_run - 1], '--', linewidth="3", color="blue")
# ax.plot3D(xx1[0, 0:loop_run-1], xx1[1, 0:loop_run-1], ss1[0:loop_run-1], linewidth = "2", color = "green")
ax.plot3D(x_e_2[0:100, 0], y_e_2[0:100, 0], ss1[0:100], linewidth="2", color="black")
ax.plot_surface(Xc_1, Yc_1, Zc_1, antialiased=False, color="blue")  # 0bstacles
ax.plot_surface(Xc_2, Yc_2, Zc_2, antialiased=False, color="blue")
ax.plot_surface(Xc_3, Yc_3, Zc_3, antialiased=False, color="blue")
ax.set_title('UAV FOLLOWS TARGET')

fig1 = mlab.figure()
mlab.clf()  # Clear the figure
mlab.plot3d(xx1[0, 0:loop_run - 1], xx1[1, 0:loop_run - 1], xx1[2, 0:loop_run - 1], tube_radius=5, color=(0, 0, 1))
mlab.plot3d(xs1[0, 0:loop_run - 1], xs1[1, 0:loop_run - 1], ss1[0:loop_run - 1], tube_radius=5, color=(1, 0, 0))
mlab.plot3d(xs1[3, 0:loop_run - 1], xs1[4, 0:loop_run - 1], ss1[0:loop_run - 1], tube_radius=5, color=(1, 0, 0))
mlab.plot3d(xs1[6, 0:loop_run - 1], xs1[7, 0:loop_run - 1], ss1[0:loop_run - 1], tube_radius=5, color=(1, 0, 0))
# mlab.plot3d(xx1[0, 0:loop_run - 1], xx1[1, 0:loop_run - 1], ss1[0:loop_run - 1], tube_radius=3, color=(0, .7, 0))
mlab.plot3d(xx1[0 + 8, 0:loop_run - 1], xx1[1 + 8, 0:loop_run - 1], xx1[2 + 8, 0:loop_run - 1], tube_radius=5,
            color=(0, 0, 1))
mlab.plot3d(xx1[0 + 8 + 8, 0:loop_run - 1], xx1[1 + 8 + 8, 0:loop_run - 1], xx1[2 + 8 + 8, 0:loop_run - 1], tube_radius=5,
            color=(0, 1, 1))
mlab.plot3d(xx1[0 + 8, 0:loop_run - 1], xx1[1 + 8, 0:loop_run - 1], ss1[0:loop_run - 1], tube_radius=3,
            color=(0.3, .7, .3))
mlab.plot3d(xx1[0, 0:loop_run - 1], xx1[1, 0:loop_run - 1], ss1[0:loop_run - 1], tube_radius=3,
            color=(0, .7, 0.7))
mlab.plot3d(xx1[0 + 8 + 8, 0:loop_run - 1], xx1[1 + 8 + 8, 0:loop_run - 1], ss1[0:loop_run - 1], tube_radius=3,
            color=(0, .7, 0))
mlab.plot3d(x_e_2[0:100, 0], y_e_2[0:100, 0], ss1[0:100], tube_radius=3, color=(0, 0, 0))
mlab.plot3d(x_e_3[0:100, 0], y_e_3[0:100, 0], ss1[0:100], tube_radius=3, color=(0, 0, 0))
mlab.plot3d(x_e_7[0:100, 0], y_e_7[0:100, 0], ss1[0:100], tube_radius=3, color=(0, 0, 0))
mlab.plot3d(x_e_1[1:loop_run - 1], y_e_1[1:loop_run - 1], ss2[1:loop_run - 1], tube_radius=3,
            color=(0.197, 0.45, 0.220))
mlab.plot3d(x_e_5[1:loop_run - 1], y_e_5[1:loop_run - 1], ss2[1:loop_run - 1], tube_radius=3, color=(0.6, 0.97, 0.28))
mlab.plot3d(x_e_6[1:loop_run - 1], y_e_6[1:loop_run - 1], ss2[1:loop_run - 1], tube_radius=3, color=(0, 0.97, 0.28))
mlab.mesh(Xc_1, Yc_1, Zc_1)
mlab.mesh(Xc_2, Yc_2, Zc_2)
mlab.mesh(Xc_3, Yc_3, Zc_3)
mlab.mesh(Xc_4, Yc_4, Zc_4)
mlab.mesh(Xc_5, Yc_5, Zc_5)
mlab.mesh(Xc_6, Yc_6, Zc_6)
mlab.mesh(Xc_7, Yc_7, Zc_7)
mlab.mesh(Xc_8, Yc_8, Zc_8)
mlab.title('Tracking UAV')
s = [0, 1900, -300, 1000, 0, 170]
mlab.orientation_axes(xlabel='X-axis', ylabel='Y-axis', zlabel='Z-axis')
# mlab.outline(color=(0, 0, 0), extent=s)
mlab.axes(color=(0, 0, 0), extent=s)
# yy = np.arange(-300,1000,100)
# zz = np.arange(0,300,10)
# y0 = np.zeros_like(yy)
# z0 = np.zeros_like(zz)
# mlab.plot3d(y0,yy,y0,line_width=5,tube_radius=5)
# mlab.plot3d(z0,z0,zz,line_width=5,tube_radius=5)
# xx = yy = zz = np.arange(0,1900,100)
# yx = np.zeros_like(xx)
# mlab.plot3d(xx, yx, yx,line_width=5,tube_radius=5)
mlab.show()

# Calculating error between UAV and FOV
error = ca.DM.zeros(loop_run + 1)
for i in range(loop_run):
    error[i] = ca.sqrt((x_e_1[i + 1] - ss[0, i]) ** 2 + (y_e_1[i + 1] - ss[1, i]) ** 2)

# print(ss.shape)
# print(error)
error_1 = ca.DM.zeros(loop_run + 1)
for i in range(loop_run):
    error_1[i] = ca.sqrt((x_e_5[i + 1] - ss[3, i]) ** 2 + (y_e_5[i + 1] - ss[4, i]) ** 2)

error_2 = ca.DM.zeros(loop_run + 1)
for i in range(loop_run):
    error_2[i] = ca.sqrt((x_e_6[i + 1] - ss[6, i]) ** 2 + (y_e_6[i + 1] - ss[7, i]) ** 2)

dist = ca.DM.zeros(loop_run + 1)
for i in range(loop_run):
    dist[i] = ca.sqrt((xx[0, i] - xx[8, i]) ** 2 + (xx[1, i] - xx[9, i]) ** 2)

dist2 = ca.DM.zeros(loop_run + 1)
for i in range(loop_run):
    dist2[i] = ca.sqrt((xx[16, i] - xx[8, i]) ** 2 + (xx[17, i] - xx[9, i]) ** 2)

dist3 = ca.DM.zeros(loop_run + 1)
for i in range(loop_run):
    dist3[i] = ca.sqrt((xx[0, i] - xx[16, i]) ** 2 + (xx[1, i] - xx[17, i]) ** 2)

dist_car_1 = ca.DM.zeros(loop_run + 1)
for i in range(loop_run):
    dist_car_1[i] = ca.sqrt((ss[0, i] - ss[3, i]) ** 2 + (ss[1, i] - ss[4, i]) ** 2)

dist_car_2 = ca.DM.zeros(loop_run + 1)
for i in range(loop_run):
    dist_car_2[i] = ca.sqrt((ss[0, i] - ss[3+3, i]) ** 2 + (ss[1, i] - ss[4+3, i]) ** 2)

dist_car_3 = ca.DM.zeros(loop_run + 1)
for i in range(loop_run):
    dist_car_3[i] = ca.sqrt((ss[6, i] - ss[3, i]) ** 2 + (ss[7, i] - ss[4, i]) ** 2)


itr = loop_run
dist = np.array(dist)
dist2 = np.array(dist2)
dist3 = np.array(dist3)
error1 = np.array(error)
error2 = np.array(error_1)
error3 = np.array(error_2)
dist_car_1 = np.array(dist_car_1)
dist_car_2 = np.array(dist_car_2)
dist_car_3 = np.array(dist_car_3)

sum_err = sum(error1[0:itr])
sum_err_1 = sum(error2[0:itr])
sum_err_2 = sum(error3[0:itr])
final_error = sum_err + sum_err_1 + sum_err_2

print(controls_uu[0, 0:max_step_size])

A1_control1 = controls_uu[0, 0:max_step_size]
A1_control2 = controls_uu[1, 0:max_step_size]
A1_control3 = controls_uu[2, 0:max_step_size]
A1_control4 = controls_uu[3, 0:max_step_size]
A1_control5 = controls_uu[4, 0:max_step_size]
A1_control6 = controls_uu[5, 0:max_step_size]
A2_control1 = controls_uu[6, 0:max_step_size]
A2_control2 = controls_uu[7, 0:max_step_size]
A2_control3 = controls_uu[8, 0:max_step_size]
A2_control4 = controls_uu[9, 0:max_step_size]
A2_control5 = controls_uu[10, 0:max_step_size]
A2_control6 = controls_uu[11, 0:max_step_size]
A3_control1 = controls_uu[12, 0:max_step_size]
A3_control2 = controls_uu[13, 0:max_step_size]
A3_control3 = controls_uu[14, 0:max_step_size]
A3_control4 = controls_uu[15, 0:max_step_size]
A3_control5 = controls_uu[16, 0:max_step_size]
A3_control6 = controls_uu[17, 0:max_step_size]

control_squared1 = np.square(A1_control1)
print(control_squared1)
control_squared2 = np.square(A1_control2)
control_squared3 = np.square(A1_control3)
control_squared4 = np.square(A1_control4)
control_squared5 = np.square(A1_control5)
control_squared6 = np.square(A1_control6)
control_squared7 = np.square(A2_control1)
control_squared8 = np.square(A2_control2)
control_squared9 = np.square(A2_control3)
control_squared10 = np.square(A2_control4)
control_squared11 = np.square(A2_control5)
control_squared12 = np.square(A2_control6)
control_squared13 = np.square(A3_control1)
control_squared14 = np.square(A3_control2)
control_squared15 = np.square(A3_control3)
control_squared16 = np.square(A3_control4)
control_squared17 = np.square(A3_control5)
control_squared18 = np.square(A3_control6)

Sum_squared1 = np.sum(control_squared1)
print(Sum_squared1)
Sum_squared2 = np.sum(control_squared2)
Sum_squared3 = np.sum(control_squared3)
Sum_squared4 = np.sum(control_squared1)
Sum_squared5 = np.sum(control_squared2)
Sum_squared6 = np.sum(control_squared3)
Sum_squared7 = np.sum(control_squared1)
Sum_squared8 = np.sum(control_squared2)
Sum_squared9 = np.sum(control_squared3)
Sum_squared10 = np.sum(control_squared1)
Sum_squared11 = np.sum(control_squared2)
Sum_squared12 = np.sum(control_squared3)
Sum_squared13 = np.sum(control_squared1)
Sum_squared14 = np.sum(control_squared2)
Sum_squared15 = np.sum(control_squared3)
Sum_squared16 = np.sum(control_squared1)
Sum_squared17 = np.sum(control_squared2)
Sum_squared18 = np.sum(control_squared3)

sqroot1 = np.sqrt(Sum_squared1)
print(sqroot1)
sqroot2 = np.sqrt(Sum_squared2)
sqroot3 = np.sqrt(Sum_squared3)
sqroot4 = np.sqrt(Sum_squared4)
sqroot5 = np.sqrt(Sum_squared5)
sqroot6 = np.sqrt(Sum_squared6)
sqroot7 = np.sqrt(Sum_squared7)
sqroot8 = np.sqrt(Sum_squared8)
sqroot9 = np.sqrt(Sum_squared9)
sqroot10 = np.sqrt(Sum_squared10)
sqroot11 = np.sqrt(Sum_squared11)
sqroot12 = np.sqrt(Sum_squared12)
sqroot13 = np.sqrt(Sum_squared13)
sqroot14 = np.sqrt(Sum_squared14)
sqroot15 = np.sqrt(Sum_squared15)
sqroot16 = np.sqrt(Sum_squared16)
sqroot17 = np.sqrt(Sum_squared17)
sqroot18 = np.sqrt(Sum_squared18)

control_effort1 = sqroot1/285.2
print(control_effort1)
control_effort2 = sqroot2/285.2
control_effort3 = sqroot3/285.2
control_effort4 = sqroot4/285.2
control_effort5 = sqroot5/285.2
control_effort6 = sqroot6/285.2
control_effort7 = sqroot7/285.2
control_effort8 = sqroot8/285.2
control_effort9 = sqroot9/285.2
control_effort10 = sqroot10/285.2
control_effort11 = sqroot11/285.2
control_effort12 = sqroot12/285.2
control_effort13 = sqroot13/285.2
control_effort14 = sqroot14/285.2
control_effort15 = sqroot15/285.2
control_effort16 = sqroot16/285.2
control_effort17 = sqroot17/285.2
control_effort18 = sqroot18/285.2

# print("Control Effort = " + str(control_effort1) + ',' + str(control_effort2) + ',' + str(control_effort3))

##################################################

# np.savez("Jo_WRL_Controls_final1.npz", controls_uu=controls_uu, error=error, error_1=error_1, error_2=error_2, xx1=xx1,
#          xs1=xs1, UAV3_FOV_Plot_WRL=UAV3_FOV_Plot_WRL, UAV2_FOV_Plot_WRL=UAV2_FOV_Plot_WRL,
#          UAV1_FOV_Plot_WRL=UAV1_FOV_Plot_WRL, control_effort1=control_effort1, control_effort2=control_effort2,
#          control_effort3=control_effort3, control_effort4=control_effort4, control_effort5=control_effort5,
#          control_effort6=control_effort6, control_effort7=control_effort7, control_effort8=control_effort8,
#          control_effort9=control_effort9, control_effort10=control_effort10, control_effort11=control_effort11,
#          control_effort12=control_effort12, control_effort13=control_effort13, control_effort14=control_effort14,
#          control_effort15=control_effort15, control_effort16=control_effort16, control_effort17=control_effort17,
#          control_effort18=control_effort18)

##################################################

# print(sum_err)
# print(final_error)
sum_err = np.squeeze(sum_err)
sum_err_1 = np.squeeze(sum_err_1)
sum_err = np.squeeze(sum_err)
sum_err_1 = np.squeeze(sum_err_2)
# final_error = np.squeeze
# print(sum_err_1.shape)
x = ["Multirotor", "FW-UAV-1", "FW-UAV-2", "Total Error"]
y = [sum_err, sum_err_1, sum_err_2, final_error]

fig = plt.figure()
plt.subplot(131)
plt.plot(error[0:itr], linewidth="2", color="red")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.subplot(132)
plt.plot(error_1[0:itr], linewidth="2", color="green")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.subplot(133)
plt.plot(error_2[0:itr], linewidth="2", color="blue")
plt.xlabel("Iteration")
plt.ylabel("Error")
# plt.legend(['Error of Drone', 'Error of FW-UAV-1', 'Error of FW-UAV-2'])

fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.plot(times1, dist[0:itr], linewidth="2", color="blue")
plt.plot(times1, dist2[0:itr], linewidth="2", color="green")
plt.plot(times1, dist3[0:itr], linewidth="2", color="red")
plt.legend(['Multirotor - FW-AAV-1', 'FW-AAV-1 - FW-AAV-2', 'FW-AAV-2 - Multirotor'])
plt.xlabel("Time (s)")
plt.ylabel("Distance b/w AAVs")

fig = plt.figure()
plt.bar(x, y)

fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.plot(dist_car_1[0:itr], linewidth="2", color="blue")
plt.plot(dist_car_2[0:itr], linewidth="2", color="green")
plt.plot(dist_car_3[0:itr], linewidth="2", color="red")
plt.legend(['Car-1 - Car-2', 'Car-2 - Car-3', 'Car-3 - Car-1'])
plt.xlabel("Time (s)")
plt.ylabel("Distance b/w Cars")


# UAV 1 Without RL
fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1, controls_uu[0, 0:max_step_size], linewidth="2", color="brown")
plt.plot(times1, linear_acc_min, '--', color="red")
plt.plot(times1, linear_acc_max, '--', color="red")
plt.title('Linear Acceleration of Multirotor')
plt.xlabel('Time (s)')
plt.ylabel('m/s')
fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1, controls_uu[1, 0:max_step_size], linewidth="2", color="blue")
plt.plot(times1, pitch_rate_min, '--', color="red")
plt.plot(times1, pitch_rate_max, '--', color="red")
plt.title('Pitch Rate of Multirotor')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')
fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1, controls_uu[2, 0:max_step_size], linewidth="2", color="black")
plt.plot(times1, yaw_rate_min, '--', color="red")
plt.plot(times1, yaw_rate_max, '--', color="red")
plt.title('Yaw Rate of Multirotor')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')
fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1, controls_uu[3, 0:max_step_size], linewidth="2", color="grey")
plt.plot(times1, groll_rate_min, '--', color="red")
plt.plot(times1, groll_rate_max, '--', color="red")
plt.title('Gimbal Roll Rate')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')
fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1, controls_uu[4, 0:max_step_size], linewidth="2", color="green")
plt.plot(times1, gpitch_rate_min, '--', color="red")
plt.plot(times1, gpitch_rate_max, '--', color="red")
plt.title('Gimbal Pitch Rate')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')
fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1, controls_uu[5, 0:max_step_size], linewidth="2", color="purple")
plt.plot(times1, gyaw_rate_min, '--', color="red")
plt.plot(times1, gyaw_rate_max, '--', color="red")
plt.title('Gimbal Yaw Rate')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')


# UAV 2 Without RL
fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1, controls_uu[0+6, 0:max_step_size], linewidth="2", color="brown")
plt.plot(times1, linear_acc_min, '--', color="red")
plt.plot(times1, linear_acc_max, '--', color="red")
plt.title('Linear Acceleration of AAV-1')
plt.xlabel('Time (s)')
plt.ylabel('m/s')
fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1, controls_uu[1+6, 0:max_step_size], linewidth="2", color="blue")
plt.plot(times1, pitch_rate_min, '--', color="red")
plt.plot(times1, pitch_rate_max, '--', color="red")
plt.title('Pitch Rate of AAV-1')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')
fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1, controls_uu[2+6, 0:max_step_size], linewidth="2", color="black")
plt.plot(times1, yaw_rate_min_1, '--', color="red")
plt.plot(times1, yaw_rate_max_1, '--', color="red")
plt.title('Yaw Rate of AAV-1')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')
fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1, controls_uu[3+6, 0:max_step_size], linewidth="2", color="grey")
plt.plot(times1, groll_rate_min, '--', color="red")
plt.plot(times1, groll_rate_max, '--', color="red")
plt.title('Gimbal Roll Rate')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')
fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1, controls_uu[4+6, 0:max_step_size], linewidth="2", color="green")
plt.plot(times1, gpitch_rate_min, '--', color="red")
plt.plot(times1, gpitch_rate_max, '--', color="red")
plt.title('Gimbal Pitch Rate')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')
fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1, controls_uu[5+6, 0:max_step_size], linewidth="2", color="purple")
plt.plot(times1, gyaw_rate_min, '--', color="red")
plt.plot(times1, gyaw_rate_max, '--', color="red")
plt.title('Gimbal Yaw Rate')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')


# UAV 3 Without RL
fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1, controls_uu[0+6+6, 0:max_step_size], linewidth="2", color="brown")
plt.plot(times1, linear_acc_min, '--', color="red")
plt.plot(times1, linear_acc_max, '--', color="red")
plt.title('Linear Acceleration of AAV-2')
plt.xlabel('Time (s)')
plt.ylabel('m/s')
fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1, controls_uu[1+6+6, 0:max_step_size], linewidth="2", color="blue")
plt.plot(times1, pitch_rate_min, '--', color="red")
plt.plot(times1, pitch_rate_max, '--', color="red")
plt.title('Pitch Rate of AAV-2')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')
fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1, controls_uu[2+6+6, 0:max_step_size], linewidth="2", color="black")
plt.plot(times1, yaw_rate_min_2, '--', color="red")
plt.plot(times1, yaw_rate_max_2, '--', color="red")
plt.title('Yaw Rate of AAV-2')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')
fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1, controls_uu[3+12, 0:max_step_size], linewidth="2", color="grey")
plt.plot(times1, groll_rate_min, '--', color="red")
plt.plot(times1, groll_rate_max, '--', color="red")
plt.title('Gimbal Roll Rate')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')
fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1, controls_uu[4+12, 0:max_step_size], linewidth="2", color="green")
plt.plot(times1, gpitch_rate_min, '--', color="red")
plt.plot(times1, gpitch_rate_max, '--', color="red")
plt.title('Gimbal Pitch Rate')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')
fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1, controls_uu[5+12, 0:max_step_size], linewidth="2", color="purple")
plt.plot(times1, gyaw_rate_min, '--', color="red")
plt.plot(times1, gyaw_rate_max, '--', color="red")
plt.title('Gimbal Yaw Rate')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')


# new velocity
fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1, xx1[24, 0:max_step_size], linewidth="2", color="brown")
plt.plot(times1, linear_vel_min, '--', color="red")
plt.plot(times1, linear_vel_max, '--', color="red")
plt.title('Linear Velocity of Multirotor')
plt.xlabel('Time (s)')
plt.ylabel('m/s')

fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1[0:max_step_size-9], xx1[25, 9:max_step_size], linewidth="2", color="brown")
plt.plot(times1[0:max_step_size-9], linear_vel_min_1[0:max_step_size-9], '--', color="red")
plt.plot(times1[0:max_step_size-9], linear_vel_max_1[0:max_step_size-9], '--', color="red")
plt.title('Linear Velocity of AAV-1')
plt.xlabel('Time (s)')
plt.ylabel('m/s')

fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1[0:max_step_size-9], xx1[26, 9:max_step_size], linewidth="2", color="brown")
plt.plot(times1[0:max_step_size-9], linear_vel_min_2[0:max_step_size-9], '--', color="red")
plt.plot(times1[0:max_step_size-9], linear_vel_max_2[0:max_step_size-9], '--', color="red")
plt.title('Linear Velocity of AAV-2')
plt.xlabel('Time (s)')
plt.ylabel('m/s')

plt.show()
