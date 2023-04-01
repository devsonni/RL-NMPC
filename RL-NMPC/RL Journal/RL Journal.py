from gym import Env
from gym.spaces import Discrete, Box, Tuple
import numpy as np
import random
import gym
from time import time
import matplotlib.pyplot as plt
import casadi as ca
from casadi import sin, cos, pi, tan
from mayavi import mlab

# from tvtk.api import tvtk
# from mayavi.modules.api import Outline

gym.logger.set_level(40)

#########################################################################
##################### MPC and Supportive functions ######################

# Global MPC variables
x0 = [100, 150, 200, 0, 0, 0, 0, 0, 100, 200, 200, 0, 0, 0, 0, 0, 0, 0]
xs = [100, 150, pi / 4, 120, 200, pi / 4, 140, 250, pi / 4]
mpc_iter = 0  # initial MPC count
max_step_size = 1426 * 2
sc = 0
ss1 = np.zeros((max_step_size + 1000))
x_o_1 = 90 + 100 + 60 + 25
y_o_1 = 400 + 20
x_o_2 = -75 + 20
y_o_2 = 500 + 20
x_o_3 = -425 - 20 + 15
y_o_3 = 400
x_o_4 = -160 - 100 + 30
y_o_4 = -10 + 50 - 30 + 20
x_o_5 = 15 + 100 - 30
y_o_5 = -10 + 50 - 30 + 20
x_o_6 = 200 + 70
y_o_6 = -330 - 30
x_o_8 = -390 - 20 + 10
y_o_8 = -330 - 30
x_o_7 = -80
y_o_7 = -480 + 20
x_o_9 = 1000
y_o_9 = 1000
x_o_10 = 1000
y_o_10 = 1000
# x_o_1 = 1000
# y_o_1 = 1000
# x_o_2 = 1000
# y_o_2 = 1000
# x_o_3 = 1000
# y_o_3 = 1000
# x_o_4 = 1000
# y_o_4 = 1000
# x_o_5 = 1000
# y_o_5 = 1000
# x_o_6 = 1000
# y_o_6 = 1000
# x_o_8 = 1000
# y_o_8 = 1000
# x_o_7 = 1000
# y_o_7 = 1000
# x_o_9 = 1000
# y_o_9 = 1000
# x_o_10 = 1000
# y_o_10 = 1000
obs_r = 30
UAV_r = 2

# Adding global variable for increasing speed of the code
# mpc parameters
T = 0.1  # discrete step
N = 5  # number of look ahead steps

# Constrains of UAV with gimbal
# input constrains of UAV
v_u_min = 14
v_u_max = 30
omega_2_u_min = - pi / 30
omega_2_u_max = pi / 30
omega_3_u_min = -7 * (pi / 21)
omega_3_u_max = 7 * (pi / 21)

v_u_min_1 = 16
v_u_max_1 = 30
omega_2_u_min_1 = - pi / 30
omega_2_u_max_1 = pi / 30
omega_3_u_min_1 = -10.5 * (pi / 21)
omega_3_u_max_1 = 10.5 * (pi / 21)


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

# states constrains of UAV
theta_u_min = -0.2618
theta_u_max = 0.2618
z_u_min = 200
z_u_max = 200

theta_u_min_1 = -0.2618
theta_u_max_1 = 0.2618
z_u_min_1 = 200
z_u_max_1 = 200

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

rel_v_u_1 = ca.SX.sym('rel_v_u_1')
rel_v_u = ca.SX.sym('rel_v_u')

a_u = ca.SX.sym('a_u')
a_u_1 = ca.SX.sym('a_u_1')

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
    v_u,
    v_u_1
)
n_states_u = states_u.numel()

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
    rel_v_u,
    rel_v_u_1
)

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
    omega_3_g_1
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
    a_u,
    a_u_1
)

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
    a_u,
    a_u_1
)

# Non-linear mapping function which is f(x,y)
f_u = ca.Function('f', [states_u, controls_u], [rhs_u])
rel_f_u = ca.Function('rel_f_u', [rel_states_u, controls_u], [rel_rhs_u])

U = ca.SX.sym('U', n_controls, N)  # Decision Variables
P = ca.SX.sym('P', n_states_u + 9 + 6 + 6)  # This consists of initial states of UAV with gimbal 1-8 and
# reference states 9-11 (reference states is target's states)

X = ca.SX.sym('X', n_states_u, (N + 1))  # Has prediction of states over prediction horizon

# Filling the defined system parameters of UAV
X[:, 0] = P[0:n_states_u]  # initial state

for k in range(N):
    st = X[:, k]
    con = U[:, k]
    f_value = f_u(st, con)
    st_next = st + T * f_value
    X[:, k + 1] = st_next

ff = ca.Function('ff', [U, P], [X])

K = 33
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
lbg[31:K * (N + 1):K] = v_u_min
lbg[32:K * (N + 1):K] = v_u_min_1

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
ubg[31:K * (N + 1):K] = v_u_max
ubg[32:K * (N + 1):K] = v_u_max_1

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

    VFOV = 0.7  # Making FOV
    HFOV = 0.7
    VFOV_1 = 0.7  # Making FOV
    HFOV_1 = 0.7

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

    obj = obj \
          + P[27] * (P[33] * ca.sqrt((stt[0, k] - P[18]) ** 2 + (stt[1, k] - P[19]) ** 2) +
                     P[34] * ((A[k] * (P[18] - X_E[k]) ** 2 + B[k] * (P[19] - Y_E[k]) * (P[18] - X_E[k])
                               + C[k] * (P[19] - Y_E[k]) ** 2) - 1) +
                     P[35] * ca.sqrt((stt[0, k] - P[21]) ** 2 + (stt[1, k] - P[22]) ** 2) +
                     P[36] * ((A[k] * (P[21] - X_E[k]) ** 2 + B[k] * (P[22] - Y_E[k]) * (P[21] - X_E[k])
                               + C[k] * (P[22] - Y_E[k]) ** 2) - 1) +
                     P[37] * ca.sqrt((stt[8, k] - P[24]) ** 2 + (stt[9, k] - P[25]) ** 2) +
                     P[38] * ((A_1[k] * (P[24] - X_E_1[k]) ** 2 + B_1[k] * (P[25] - Y_E_1[k]) * (P[24] - X_E_1[k])
                               + C_1[k] * (P[25] - Y_E_1[k]) ** 2) - 1)) \
          + P[28] * (P[33] * ca.sqrt((stt[0, k] - P[18]) ** 2 + (stt[1, k] - P[19]) ** 2) +
                     P[34] * ((A[k] * (P[18] - X_E[k]) ** 2 + B[k] * (P[19] - Y_E[k]) * (P[18] - X_E[k])
                               + C[k] * (P[19] - Y_E[k]) ** 2) - 1) +
                     P[35] * ca.sqrt((stt[8, k] - P[21]) ** 2 + (stt[9, k] - P[22]) ** 2) +
                     P[36] * ((A_1[k] * (P[21] - X_E_1[k]) ** 2 + B_1[k] * (P[22] - Y_E_1[k]) * (P[21] - X_E_1[k])
                               + C_1[k] * (P[22] - Y_E_1[k]) ** 2) - 1) +
                     P[37] * ca.sqrt((stt[8, k] - P[24]) ** 2 + (stt[9, k] - P[25]) ** 2) +
                     P[38] * ((A_1[k] * (P[24] - X_E_1[k]) ** 2 + B_1[k] * (P[25] - Y_E_1[k]) * (P[24] - X_E_1[k])
                               + C_1[k] * (P[25] - Y_E_1[k]) ** 2) - 1)) \
          + P[29] * (P[33] * ca.sqrt((stt[0, k] - P[21]) ** 2 + (stt[1, k] - P[22]) ** 2) +
                     P[34] * ((A[k] * (P[21] - X_E[k]) ** 2 + B[k] * (P[22] - Y_E[k]) * (P[21] - X_E[k])
                               + C[k] * (P[22] - Y_E[k]) ** 2) - 1) +
                     P[35] * ca.sqrt((stt[8, k] - P[18]) ** 2 + (stt[9, k] - P[19]) ** 2) +
                     P[36] * ((A_1[k] * (P[18] - X_E_1[k]) ** 2 + B_1[k] * (P[19] - Y_E_1[k]) * (P[18] - X_E_1[k])
                               + C_1[k] * (P[19] - Y_E_1[k]) ** 2) - 1) +
                     P[37] * ca.sqrt((stt[8, k] - P[24]) ** 2 + (stt[9, k] - P[25]) ** 2) +
                     P[38] * ((A_1[k] * (P[24] - X_E_1[k]) ** 2 + B_1[k] * (P[25] - Y_E_1[k]) * (P[24] - X_E_1[k])
                               + C_1[k] * (P[25] - Y_E_1[k]) ** 2) - 1)) \
          + P[30] * (P[33] * ca.sqrt((stt[0, k] - P[24]) ** 2 + (stt[1, k] - P[25]) ** 2) +
                     P[34] * ((A[k] * (P[24] - X_E[k]) ** 2 + B[k] * (P[25] - Y_E[k]) * (P[24] - X_E[k])
                               + C[k] * (P[25] - Y_E[k]) ** 2) - 1) +
                     P[35] * ca.sqrt((stt[8, k] - P[18]) ** 2 + (stt[9, k] - P[19]) ** 2) +
                     P[36] * ((A_1[k] * (P[18] - X_E_1[k]) ** 2 + B_1[k] * (P[19] - Y_E_1[k]) * (P[18] - X_E_1[k])
                               + C_1[k] * (P[19] - Y_E_1[k]) ** 2) - 1) +
                     P[37] * ca.sqrt((stt[8, k] - P[21]) ** 2 + (stt[9, k] - P[22]) ** 2) +
                     P[38] * ((A_1[k] * (P[21] - X_E_1[k]) ** 2 + B_1[k] * (P[22] - Y_E_1[k]) * (P[21] - X_E_1[k])
                               + C_1[k] * (P[22] - Y_E_1[k]) ** 2) - 1)) \
          + P[31] * (P[33] * ca.sqrt((stt[0, k] - P[21]) ** 2 + (stt[1, k] - P[22]) ** 2) +
                     P[34] * ((A[k] * (P[21] - X_E[k]) ** 2 + B[k] * (P[22] - Y_E[k]) * (P[21] - X_E[k])
                               + C[k] * (P[22] - Y_E[k]) ** 2) - 1) +
                     P[35] * ca.sqrt((stt[0, k] - P[24]) ** 2 + (stt[1, k] - P[25]) ** 2) +
                     P[36] * ((A[k] * (P[24] - X_E[k]) ** 2 + B[k] * (P[25] - Y_E[k]) * (P[24] - X_E[k])
                               + C[k] * (P[25] - Y_E[k]) ** 2) - 1) +
                     P[37] * ca.sqrt((stt[8, k] - P[18]) ** 2 + (stt[9, k] - P[19]) ** 2) +
                     P[38] * ((A_1[k] * (P[18] - X_E_1[k]) ** 2 + B_1[k] * (P[19] - Y_E_1[k]) * (P[18] - X_E_1[k])
                               + C_1[k] * (P[19] - Y_E_1[k]) ** 2) - 1)) \
          + P[32] * (P[33] * ca.sqrt((stt[0, k] - P[18]) ** 2 + (stt[1, k] - P[19]) ** 2) +
                     P[34] * ((A[k] * (P[18] - X_E[k]) ** 2 + B[k] * (P[19] - Y_E[k]) * (P[18] - X_E[k])
                               + C[k] * (P[19] - Y_E[k]) ** 2) - 1) +
                     P[35] * ca.sqrt((stt[0, k] - P[24]) ** 2 + (stt[1, k] - P[25]) ** 2) +
                     P[36] * ((A[k] * (P[24] - X_E[k]) ** 2 + B[k] * (P[25] - Y_E[k]) * (P[24] - X_E[k])
                               + C[k] * (P[25] - Y_E[k]) ** 2) - 1) +
                     P[37] * ca.sqrt((stt[8, k] - P[30]) ** 2 + (stt[9, k] - P[31]) ** 2) +
                     P[38] * ((A_1[k] * (P[21] - X_E_1[k]) ** 2 + B_1[k] * (P[22] - Y_E_1[k]) * (P[21] - X_E_1[k])
                               + C_1[k] * (P[22] - Y_E_1[k]) ** 2) - 1))

# compute the constraints, states or inequality constrains
# compute the constraints, states or inequality constrains
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
                   -ca.sqrt((X[0, k] - x_o_10) ** 2 + (X[1, k] - y_o_10) ** 2) + (UAV_r + obs_r),
                   # limit of obstacle-10
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
                   -ca.sqrt((X[8, k] - X[0, k]) ** 2 + (X[9, k] - X[1, k]) ** 2) + 2,  # collusion with each other
                   X[16, k],
                   X[17, k]
                   )

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
# solver.stats()

args = {
    'lbg': lbg,  # lower bound for state
    'ubg': ubg,  # upper bound for state
    'lbx': lbx,  # lower bound for controls
    'ubx': ubx  # upper bound for controls
}


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
    global sc
    v = 12
    v_1 = 12
    v_2 = 12

    con_t = [v, 3 * pi / 200, v_1, 3 * pi / 200, v_2, 3 * pi / 200]  # right turn
    if (sc >= 2 * (500)):
        con_t = [v, 0, v_1, 0, v_2, 0]  # left ahead
    if (sc >= 2 * (700 + 10 + 1 + 1)):
        con_t = [v, -3 * pi / 200, v_1, -3 * pi / 200, v_2, -3 * pi / 200]  # 50 ahead
    if (sc >= 2 * (1200 + 10 + 1 + 1)):
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


# need some modification
def max_index(arr):
    shape = np.shape(arr)
    max_indices = (0, 0)
    for i in range(shape[0]):
        for k in range(shape[1]):
            if arr[max_indices] < arr[i, k]:
                max_indices = (i, k)
    return max_indices


def max_index_3(arr):
    shape = np.shape(arr)
    max_indices = (0, 0, 0)
    for i in range(shape[0]):
        for k in range(shape[1]):
            for j in range(shape[2]):
                if arr[max_indices] < arr[i, k, j]:
                    max_indices = (i, k, j)
    return max_indices


# main MPC function
def MPC(w1, w2, w3, w4, w5, w6, w7):
    global T, N, v_u_min, v_u_max, omega_2_u_min, omega_2_u_max, omega_3_u_min, omega_3_u_max, omega_1_g_min, \
        omega_1_g_max, omega_2_g_min, omega_2_g_max, omega_3_g_min, omega_3_g_max, theta_u_min, theta_u_max, z_u_min, \
        z_u_max, phi_g_min, phi_g_max, theta_g_min, theta_g_max, shi_g_min, shi_g_max, x_o_1, y_o_1, obs_r, x_o_2, \
        y_o_2, x_o_3, y_o_3, x_u, y_u, z_u, theta_u, psi_u, phi_g, shi_g, theta_g, states_u, n_states_u, v_u, omega_2_u, \
        omega_3_u, omega_1_g, omega_2_g, omega_3_g, controls_u, n_controls, rhs_u, f_u, U, P, X, lbg, ubg, lbx, ubx, ff, \
        obj, g, OPT_variables, nlp_prob, opts, solver, args, x0, xs, mpc_iter, sc, \
        v_u_min_1, v_u_max_1, omega_2_u_min_1, omega_2_u_max_1, omega_3_u_min_1, omega_3_u_max_1, omega_1_g_min_1, \
        omega_1_g_max_1, omega_2_g_min_1, omega_2_g_max_1, omega_3_g_min_1, omega_3_g_max_1, theta_u_min_1, \
        theta_u_max_1, z_u_min_1, z_u_max_1, phi_g_min_1, phi_g_max_1, theta_g_min_1, theta_g_max_1, shi_g_min_1, \
        shi_g_max_1, x_u_1, y_u_1, z_u_1, theta_u_1, psi_u_1, phi_g_1, shi_g_1, theta_g_1, v_u_1, omega_2_u_1, \
        omega_3_u_1, omega_1_g_1, omega_2_g_1, omega_3_g_1, \
        v_u_min_2, v_u_max_2, omega_2_u_min_2, omega_2_u_max_2, omega_3_u_min_2, omega_3_u_max_2, omega_1_g_min_2, \
        omega_1_g_max_2, omega_2_g_min_2, omega_2_g_max_2, omega_3_g_min_2, omega_3_g_max_2, theta_u_min_2, \
        theta_u_max_2, z_u_min_2, z_u_max_2, phi_g_min_2, phi_g_max_2, theta_g_min_2, \
        theta_g_max_2, shi_g_min_2, shi_g_max_2, x_u_2, y_u_2, z_u_2, theta_u_2, psi_u_2, phi_g_2, shi_g_2, theta_g_2, \
        v_u_2, omega_2_u_2, omega_3_u_2, omega_1_g_2, omega_2_g_2, omega_3_g_2, rel_f_u

    t1 = time()
    t0 = 0

    if (w1 == 1):
        m1, m2, m3, m4, m5, m6 = 1, 0, 0, 0, 0, 0
    elif (w1 == 2):
        m1, m2, m3, m4, m5, m6 = 0, 1, 0, 0, 0, 0
    elif (w1 == 3):
        m1, m2, m3, m4, m5, m6 = 0, 0, 1, 0, 0, 0
    elif (w1 == 4):
        m1, m2, m3, m4, m5, m6 = 0, 0, 0, 1, 0, 0
    elif (w1 == 5):
        m1, m2, m3, m4, m5, m6 = 0, 0, 0, 0, 1, 0
    elif (w1 == 6):
        m1, m2, m3, m4, m5, m6 = 0, 0, 0, 0, 0, 1

    # xx = DM(state_init)
    t = ca.DM(t0)

    u0 = ca.DM.zeros((n_controls, N))
    xx = ca.DM.zeros((n_states_u, max_step_size + 1))
    # ss = ca.DM.zeros((3, 801))

    x_e_1 = ca.DM.zeros((max_step_size + 1))
    y_e_1 = ca.DM.zeros((max_step_size + 1))
    x_e_5 = ca.DM.zeros((max_step_size + 1))
    y_e_5 = ca.DM.zeros((max_step_size + 1))
    x_e_6 = ca.DM.zeros((max_step_size + 1))
    y_e_6 = ca.DM.zeros((max_step_size + 1))

    # xx[:, 0] = x0
    # ss[:, 0] = xs
    mpc_iter = 0

    times = np.array([[0]])

    ###############################################################################
    ##############################   Main Loop    #################################

    ss[:, mpc_iter] = xs
    args['p'] = ca.vertcat(
        x0,  # current state
        xs,
        m1,
        m2,
        m3,
        m4,
        m5,
        m6,
        w2,
        w3,
        w4,
        w5,
        w6,
        w7  # target state
    )

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

    t0, x0, u0, xs = shift_timestep(T, t0, x0, u, rel_f_u, xs)

    # tracking states of target and UAV for plotting
    xx[:, mpc_iter + 1] = x0

    # xx ...
    t2 = time()
    # print(t2-t1)
    times = np.vstack((
        times,
        t2 - t1
    ))

    sc = sc + 1
    mpc_iter = mpc_iter + 1

    a_p = (x0[2] * (tan(x0[6] + VFOV / 2)) - x0[2] * tan(x0[6] - VFOV / 2)) / 2  # For plotting FOV
    b_p = (x0[2] * (tan(x0[5] + HFOV / 2)) - x0[2] * tan(x0[5] - HFOV / 2)) / 2
    x_e_1[mpc_iter] = x0[0] + a_p + x0[2] * (tan(x0[6] - VFOV / 2))
    y_e_1[mpc_iter] = x0[1] + b_p + x0[2] * (tan(x0[5] - HFOV / 2))

    a_p_1 = (x0[10] * (tan(x0[14] + VFOV_1 / 2)) - x0[10] * tan(x0[14] - VFOV_1 / 2)) / 2
    b_p_1 = (x0[10] * (tan(x0[13] + HFOV_1 / 2)) - x0[10] * tan(x0[13] - HFOV_1 / 2)) / 2
    x_e_5[mpc_iter] = x0[8] + a_p_1 + x0[10] * (tan(x0[14] - VFOV_1 / 2))
    y_e_5[mpc_iter] = x0[9] + b_p_1 + x0[10] * (tan(x0[13] - HFOV_1 / 2))

    # error function will be selected according to mode selection
    if (w1 == 1):
        Error0 = ca.sqrt((x_e_1[mpc_iter] -
                          ((ss[0, mpc_iter - 1] + ss[3, mpc_iter - 1]) / 2)) ** 2 +
                         ((y_e_1[mpc_iter] - ((ss[1, mpc_iter - 1] + ss[4, mpc_iter - 1]) / 2)) ** 2))
        Error1 = ca.sqrt((x_e_5[mpc_iter] - ss[6, mpc_iter - 1]) ** 2 + (y_e_5[mpc_iter] - ss[7, mpc_iter - 1]) ** 2)
        # Error2 = ca.sqrt((x_e_6[mpc_iter] - ss[6, mpc_iter - 1]) ** 2 + (y_e_6[mpc_iter] - ss[7, mpc_iter - 1]) ** 2)
    elif (w1 == 2):
        Error0 = ca.sqrt((x_e_1[mpc_iter] - ss[0, mpc_iter - 1]) ** 2 + (y_e_1[mpc_iter] - ss[1, mpc_iter - 1]) ** 2)
        Error1 = ca.sqrt((x_e_5[mpc_iter] -
                          ((ss[3, mpc_iter - 1] + ss[6, mpc_iter - 1]) / 2)) ** 2 +
                         (y_e_5[mpc_iter] - ((ss[4, mpc_iter - 1] + ss[7, mpc_iter - 1]) / 2)) ** 2)
        # Error2 = ca.sqrt((x_e_6[mpc_iter] - ss[6, mpc_iter - 1]) ** 2 + (y_e_6[mpc_iter] - ss[7, mpc_iter - 1]) ** 2)
    elif (w1 == 3):
        Error0 = ca.sqrt((x_e_1[mpc_iter] - ss[3, mpc_iter - 1]) ** 2 + (y_e_1[mpc_iter] - ss[4, mpc_iter - 1]) ** 2)
        Error1 = ca.sqrt((x_e_5[mpc_iter] -
                          ((ss[0, mpc_iter - 1] + ss[6, mpc_iter - 1]) / 2)) ** 2 +
                         (y_e_5[mpc_iter] - ((ss[1, mpc_iter - 1] + ss[7, mpc_iter - 1]) / 2)) ** 2)
        # Error2 = ca.sqrt((x_e_6[mpc_iter] - ss[6, mpc_iter - 1]) ** 2 + (y_e_6[mpc_iter] - ss[7, mpc_iter - 1]) ** 2)
    elif (w1 == 4):
        Error0 = ca.sqrt((x_e_1[mpc_iter] - ss[6, mpc_iter - 1]) ** 2 + (y_e_1[mpc_iter] - ss[7, mpc_iter - 1]) ** 2)
        Error1 = ca.sqrt((x_e_5[mpc_iter] -
                          ((ss[0, mpc_iter - 1] + ss[3, mpc_iter - 1]) / 2)) ** 2 +
                         (y_e_5[mpc_iter] - ((ss[1, mpc_iter - 1] + ss[4, mpc_iter - 1]) / 2)) ** 2)
        # Error2 = ca.sqrt((x_e_6[mpc_iter] - ss[6, mpc_iter - 1]) ** 2 + (y_e_6[mpc_iter] - ss[7, mpc_iter - 1]) ** 2)
    elif (w1 == 5):
        Error0 = ca.sqrt((x_e_1[mpc_iter] -
                          ((ss[6, mpc_iter - 1] + ss[3, mpc_iter - 1]) / 2)) ** 2 +
                         ((y_e_1[mpc_iter] - ((ss[7, mpc_iter - 1] + ss[4, mpc_iter - 1]) / 2)) ** 2))
        Error1 = ca.sqrt((x_e_5[mpc_iter] - ss[0, mpc_iter - 1]) ** 2 + (y_e_5[mpc_iter] - ss[1, mpc_iter - 1]) ** 2)
        # Error2 = ca.sqrt((x_e_6[mpc_iter] - ss[6, mpc_iter - 1]) ** 2 + (y_e_6[mpc_iter] - ss[7, mpc_iter - 1]) ** 2)
    elif (w1 == 6):
        Error0 = ca.sqrt((x_e_1[mpc_iter] -
                          ((ss[0, mpc_iter - 1] + ss[6, mpc_iter - 1]) / 2)) ** 2 +
                         ((y_e_1[mpc_iter] - ((ss[1, mpc_iter - 1] + ss[7, mpc_iter - 1]) / 2)) ** 2))
        Error1 = ca.sqrt((x_e_5[mpc_iter] - ss[3, mpc_iter - 1]) ** 2 + (y_e_5[mpc_iter] - ss[4, mpc_iter - 1]) ** 2)
        # Error2 = ca.sqrt((x_e_6[mpc_iter] - ss[6, mpc_iter - 1]) ** 2 + (y_e_6[mpc_iter] - ss[7, mpc_iter - 1]) ** 2)

    Error3 = Error0 + Error1

    return Error0, Error1, Error3, x0[0:n_states_u], xs, a_p, b_p, x_e_1[mpc_iter], y_e_1[mpc_iter], a_p_1, b_p_1, \
           x_e_5[mpc_iter], y_e_5[mpc_iter]


#################################################################################
############# Defining "tunning" reinforcement learning environment #############

class Tunning(Env):
    def __init__(self):
        # Action space which contains two discrete actions which are weights of MPC
        self.action_space = Tuple(
            spaces=(Discrete(7), Discrete(101), Discrete(101), Discrete(101), Discrete(101), Discrete(101),
                    Discrete(101)))  # This is for selecting action for the RL agent means this is action space...
        # where first value is for selecting mode of control and rest of the values are for tuning particular mode's UAV and target pair
        # limit vector of observation, observation space is [x, y, z] position of UAV
        high = np.array(
            [
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.observation_space = Box(-high, high, dtype=np.float32)
        # Episode length
        global max_step_size
        self.episode_length = max_step_size

    def step(self, action):

        # Reduce episode length by 1 step
        self.episode_length -= 1

        # Calculate reward
        error, error1, error3, obs, obs2, a_p, b_p, x_e, y_e, a_p_1, b_p_1, x_e_1, y_e_1 \
            = MPC(action[0], action[1], action[2], action[3], action[4], action[5], action[6])

        # reward = 1/(error + error3)
        reward = 1 / error
        reward1 = 1 / error1

        # Check if episode is done or not
        if self.episode_length <= 0:
            done = True
        else:
            done = False

        # extra info about env
        info = {}

        # Return step information
        return obs, obs2, reward, reward1, done, info, error, error1, error3, a_p, b_p, x_e, y_e, \
               a_p_1, b_p_1, x_e_1, y_e_1

    def render(self):
        # Implement viz
        pass

    def reset(self):
        # Reset UAV & target to initial position
        global x0, xs, mpc_iter, max_step_size, ss, sc
        x0 = [100, 150, 200, 0, 0, 0, 0, 0, 100, 200, 200, 0, 0, 0, 0, 0, 0, 0]
        xs = [100, 150, pi / 4, 120, 200, pi / 4, 140, 250, pi / 4]
        sc = 0
        mpc_iter = 0
        ss = ca.DM.zeros((3 + 3 + 3, max_step_size))
        self.episode_length = max_step_size


env = Tunning()

#################################################################
########################### Q-learning ##########################
step_size = max_step_size + 1  # Change according to main loop run
qtable = np.zeros((step_size, 101, 101))
qtable1 = np.zeros((step_size, 101, 101, 101))
qtable2 = np.zeros((step_size, 101, 101, 101))

# Q - learning parameters
total_episodes = 500  # 50 Total episodes
learning_rate = 0.9  # Learning rate 0.8 is good
max_steps = max_step_size  # Max steps per episode
gamma = 0.8  # Discounting rate 0.1 is good

# Exploration parameters
epsilon = 1.0  # Exploration rate
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.2  # Minimum exploration probability
decay_rate = 0.009  # Exponential decay rate for exploration prob

# List and array of rewards, errors etc.
rewards = []
rewardarr = np.zeros((total_episodes, max_steps))
errorarr = np.zeros((total_episodes, max_steps))
erroravg = np.zeros((total_episodes))
rewardavg = np.zeros((total_episodes))
maxreward = np.zeros((max_steps))
UAV1_RL_Err = np.zeros((max_steps))
UAV2_RL_Err = np.zeros((max_steps))
UAV1_cen = np.zeros((max_steps,2))
UAV2_cen = np.zeros((max_steps,2))
UAV3_RL_Err = np.zeros((max_steps))
UAV1_RL_Err_1 = np.zeros((max_steps))
UAV2_RL_Err_1 = np.zeros((max_steps))
UAV3_RL_Err_1 = np.zeros((max_steps))
UAV1_WRL_Err = np.zeros((max_steps))
UAV2_WRL_Err = np.zeros((max_steps))
UAV3_WRL_Err = np.zeros((max_steps))
minerrorarr = np.zeros((max_steps))
minerrorarr_1 = np.zeros((max_steps))
action_str = np.zeros((max_steps + 1, 2))
# testing 7 dimensional array (qtable)
# qtable5 = np.zeros((step_size, 7, 101, 101, 101, 101, 101, 101))
w_1 = np.zeros((max_steps, total_episodes))
w_2 = np.zeros((max_steps, total_episodes))
w_3 = np.zeros((max_steps, total_episodes))
w_4 = np.zeros((max_steps, total_episodes))
w_5 = np.zeros((max_steps, total_episodes))
w_6 = np.zeros((max_steps, total_episodes))
w_7 = np.zeros((max_steps, total_episodes))
UAV_RL = ca.DM.zeros((n_states_u, max_step_size + 1))
UAV_RL_1 = ca.DM.zeros((n_states_u, max_step_size + 1))
UAV_W_RL = ca.DM.zeros((n_states_u, max_step_size + 1))
targetarr = ca.DM.zeros((9, max_step_size + 1))
error_w_rl = np.zeros((max_steps + 1))

# Creating x axis and y axis for 3-d surface
x_s = np.zeros((total_episodes))
y_s = np.zeros((max_steps))

# Plotting calculations
for i in range(total_episodes):
    x_s[i] = i + 1

for i in range(max_steps):
    y_s[i] = i + 1

x, y = np.meshgrid(x_s, y_s)

# 2 For life or until learning is stopped
for episode in range(total_episodes):
    # Reset the environment
    # global x0, xs
    print('\n-------- We are in episode {}, {} steps will be run --------\n'.format(episode + 1, max_steps))

    env.reset()
    step = 0
    done = False
    total_rewards = 0
    # print(qtable)

    while not done:
        # 3. Choose an action a in the current world state (s)
        ## First we randomize a number
        exp_exp_tradeoff = random.uniform(0, 1)

        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            action1 = max_index(qtable[step, :, :])
            action2 = max_index_3(qtable1[step, :, :, :])
            action3 = max_index_3(qtable2[step, :, :, :])
            action = (action1[0], action2[0], action2[1], action2[2], action3[0], action3[1], action3[2])
            print("Exploit")

        # Else doing a random choice --> exploration
        else:
            action = (0, 0, 0, 0, 0, 0, 0)
            # print("exploit")
            while (action[0] == 0 or action[1] == 0 or action[2] == 0 or action[3] == 0 or action[4] == 0 or
                   action[5] == 0 or action[6] == 0):
                action = env.action_space.sample()

        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, obs2, reward, reward1, done, info, error, error1, error3, a_p, b_p, x_e, y_e, a_p_1, \
        b_p_1, x_e_1, y_e_1 = env.step(action)

        reward_t = reward + reward1

        # action[1] = temp
        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        # qtable[new_state,:] : all the actions we can take from new state

        # This is for mode
        qtable[step, action[0], 1] = (1 - learning_rate) * (qtable[step, action[0], 1]) + \
                                     learning_rate * (reward_t + gamma * np.max(qtable[step + 1, :, :]))

        qtable1[step, action[1], action[2], action[3]] = (1 - learning_rate) * \
                                                         (qtable1[step, action[1], action[2], action[3]]) + \
                                                         learning_rate * (reward_t + gamma * np.max(qtable1[step + 1, :, :, :]))

        qtable2[step, action[4], action[5], action[6]] = (1 - learning_rate) * \
                                                         (qtable2[step, action[4], action[5], action[6]]) + \
                                                         learning_rate * (reward_t + gamma * np.max(qtable2[step + 1, :, :, :]))


        print("step {}".format(step + 1))
        # print(qtable[step, :, :])
        # print(action[0],action[1])

        step += 1
        total_rewards += reward_t
        rewardarr[episode, step - 1] = reward_t
        errorarr[episode, step - 1] = error3

        w_1[step - 1, episode] = action[0]
        w_2[step - 1, episode] = action[1]
        w_3[step - 1, episode] = action[2]
        w_4[step - 1, episode] = action[3]
        w_5[step - 1, episode] = action[4]
        w_6[step - 1, episode] = action[5]
        w_7[step - 1, episode] = action[6]

    erroravg[episode] = (sum(errorarr[episode, :]))
    rewardavg[episode] = (sum(rewardarr[episode, :]))
    print(sum(errorarr[episode, :]))

    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    rewards.append(total_rewards)

# Optimal and Initial Plotting
Error = 0
env.reset()
UAV_RL[:, 0] = x0[0:n_states_u]
targetarr[:, 0] = xs
# Printing Optimal Policy
for i in range(max_step_size):
    action1 = max_index(qtable[i, :, :])
    action2 = max_index_3(qtable1[i, :, :, :])
    action3 = max_index_3(qtable2[i, :, :, :])
    action = (action1[0], action2[0], action2[1], action2[2], action3[0], action3[1], action3[2])
    new_state, obs2, reward, reward1, done, info, error, error1, error3, a_p, b_p, x_e, y_e, a_p_1, \
    b_p_1, x_e_1, y_e_1 = env.step(action)
    UAV1_cen[i, 0] = x_e
    UAV1_cen[i, 1] = y_e
    UAV2_cen[i, 0] = x_e_1
    UAV2_cen[i, 1] = y_e_1
    UAV1_RL_Err[i] = error
    UAV2_RL_Err[i] = error1
    UAV_RL[:, i + 1] = new_state
    targetarr[:, i + 1] = obs2
    maxreward[i] = reward
    minerrorarr[i] = error3
    Error += error3
    print(i)
    # print(max_index(qtable[i, :, :]))

dist = ca.DM.zeros(max_step_size + 1)
for i in range(max_step_size):
    dist[i] = ca.sqrt((UAV_RL[0, i] - UAV_RL[8, i]) ** 2 + (UAV_RL[1, i] - UAV_RL[9, i]) ** 2)

dist2 = ca.DM.zeros(max_step_size + 1)
for i in range(max_step_size):
    dist2[i] = ca.sqrt((UAV_RL[16, i] - UAV_RL[8, i]) ** 2 + (UAV_RL[17, i] - UAV_RL[9, i]) ** 2)

dist3 = ca.DM.zeros(max_step_size + 1)
for i in range(max_step_size):
    dist3[i] = ca.sqrt((UAV_RL[0, i] - UAV_RL[16, i]) ** 2 + (UAV_RL[1, i] - UAV_RL[17, i]) ** 2)

dist = np.array(dist)
dist2 = np.array(dist2)
dist3 = np.array(dist3)

x_e_rl, y_e_rl = ellipse(a_p, b_p, x_e, y_e)
x_e_rl = np.array(x_e_rl)
y_e_rl = np.array(y_e_rl)
x_e_rl_1, y_e_rl_1 = ellipse(a_p_1, b_p_1, x_e_1, y_e_1)
x_e_rl_1 = np.array(x_e_rl_1)
y_e_rl_1 = np.array(y_e_rl_1)
print('Error of Optimal Policy: {}'.format(Error))

#############################################################
#############################################################

# for without RL error and trajectory
Error = 0
env.reset()
UAV_W_RL[:, 0] = x0[0:n_states_u]
# Printing Optimal Policy
for i in range(max_step_size):
    action = (1, 1, 1, 1, 1, 1, 1)
    # mpc_iter = mpc_iter + 1
    new_state, obs2, reward, reward1, done, info, error, error1, error3, a_p, b_p, x_e, y_e, a_p_1, \
    b_p_1, x_e_1, y_e_1 = env.step(action)
    UAV1_WRL_Err[i] = error
    UAV2_WRL_Err[i] = error1
    # UAV3_WRL_Err[i] = error2
    UAV_W_RL[:, i + 1] = new_state
    error_w_rl[i] = error3
    Error += error3
    print(i)

dist4 = ca.DM.zeros(max_step_size + 1)
for i in range(max_step_size):
    dist4[i] = ca.sqrt((UAV_W_RL[0, i] - UAV_W_RL[8, i]) ** 2 + (UAV_W_RL[1, i] - UAV_W_RL[9, i]) ** 2)

dist5 = ca.DM.zeros(max_step_size + 1)
for i in range(max_step_size):
    dist5[i] = ca.sqrt((UAV_W_RL[16, i] - UAV_W_RL[8, i]) ** 2 + (UAV_W_RL[17, i] - UAV_W_RL[9, i]) ** 2)

dist6 = ca.DM.zeros(max_step_size + 1)
for i in range(max_step_size):
    dist6[i] = ca.sqrt((UAV_W_RL[0, i] - UAV_W_RL[16, i]) ** 2 + (UAV_W_RL[1, i] - UAV_W_RL[17, i]) ** 2)

dist4 = np.array(dist4)
dist5 = np.array(dist5)
dist6 = np.array(dist6)

x_e_w_rl, y_e_w_rl = ellipse(a_p, b_p, x_e, y_e)
x_e_w_rl = np.array(x_e_w_rl)
y_e_w_rl = np.array(y_e_w_rl)
x_e_w_rl_1, y_e_w_rl_1 = ellipse(a_p_1, b_p_1, x_e_1, y_e_1)
x_e_w_rl_1 = np.array(x_e_w_rl_1)
y_e_w_rl_1 = np.array(y_e_w_rl_1)
print('Error without RL: {}'.format(Error))

################################################################
################### Saving all required arrays #################

np.savez("Journal_500_09_08_009_del.npz", errorarr=errorarr, w_1=w_1, w_2=w_2, w_3=w_3, w_4=w_4, w_5=w_5, w_6=w_6,
         erroravg=erroravg, w_7=w_7, rewardavg=rewardavg, rewardarr=rewardarr, qtable=qtable, qtable2=qtable2,
         qtable1=qtable1, total_episodes=total_episodes, max_step_size=max_step_size)

# Collecing data of obstacles for plotting cylinders
Xc_1, Yc_1, Zc_1 = data_for_cylinder_along_z(x_o_1, y_o_1, obs_r, 250)
Xc_2, Yc_2, Zc_2 = data_for_cylinder_along_z(x_o_2, y_o_2, obs_r, 250)
Xc_3, Yc_3, Zc_3 = data_for_cylinder_along_z(x_o_3, y_o_3, obs_r, 250)
Xc_4, Yc_4, Zc_4 = data_for_cylinder_along_z(x_o_4, y_o_4, obs_r, 250)
Xc_5, Yc_5, Zc_5 = data_for_cylinder_along_z(x_o_5, y_o_5, obs_r, 250)
Xc_6, Yc_6, Zc_6 = data_for_cylinder_along_z(x_o_6, y_o_6, obs_r, 250)
Xc_7, Yc_7, Zc_7 = data_for_cylinder_along_z(x_o_7, y_o_7, obs_r, 250)
Xc_8, Yc_8, Zc_8 = data_for_cylinder_along_z(x_o_8, y_o_8, obs_r, 250)
Xc_9, Yc_9, Zc_9 = data_for_cylinder_along_z(x_o_9, y_o_9, obs_r, 250)
Xc_10, Yc_10, Zc_10 = data_for_cylinder_along_z(x_o_10, y_o_10, obs_r, 250)

################################################################
############# Plotting the reward & printing actions ###########
last_action = action_str[1:51, 0:2]
sum_without_rl = sum(error_w_rl)
sum_first_episode = sum(errorarr[0, 0:max_steps])
sum_last_episode = sum(errorarr[total_episodes - 1, 0:max_steps])
sum_optimal_policy = sum(minerrorarr)
sum_optimal_policy_1 = sum(minerrorarr_1)
x_axis_bar = ['Error without RL', 'Error of FirstEpisode', 'Error of LastEpisode', 'Error of OptimalPolicy',
              'Error of OptimalPolicy 1']
y_axis_bar = [sum_without_rl, sum_first_episode, sum_last_episode, sum_optimal_policy, sum_optimal_policy_1]

sum_error_uav1_rl = sum(UAV1_RL_Err)
sum_error_uav2_rl = sum(UAV2_RL_Err)
sum_error_uav3_rl = sum(UAV3_RL_Err)
sum_error_uav1_rl_1 = sum(UAV1_RL_Err_1)
sum_error_uav2_rl_1 = sum(UAV2_RL_Err_1)
sum_error_uav3_rl_1 = sum(UAV3_RL_Err_1)
sum_error_uav1_wrl = sum(UAV1_WRL_Err)
sum_error_uav2_wrl = sum(UAV2_WRL_Err)
sum_error_uav3_wrl = sum(UAV3_WRL_Err)

x_ax_uav1 = ['Error without RL', 'Error of OptimalPolicy', 'Error of OptimalPolicy 1']
y_ax_uav1 = [sum_error_uav1_wrl, sum_error_uav1_rl, sum_error_uav1_rl_1]

x_ax_uav2 = ['Error without RL', 'Error of OptimalPolicy', 'Error of OptimalPolicy 1']
y_ax_uav2 = [sum_error_uav2_wrl, sum_error_uav2_rl, sum_error_uav2_rl_1]

x_ax_uav3 = ['Error without RL', 'Error of OptimalPolicy', 'Error of OptimalPolicy 1']
y_ax_uav3 = [sum_error_uav3_wrl, sum_error_uav3_rl, sum_error_uav3_rl_1]

x_ax_uav_f = ['Total Error without RL', 'Total Error of OptimalPolicy', 'Total Error of OptimalPolicy 1']
y_ax_uav_f = [sum_without_rl, sum_optimal_policy, sum_optimal_policy_1]

# Printing actions
# print(last_action)
my_cmap = plt.get_cmap('cool')

# plotting actions evolving over episodes for w_1
fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(x, y, w_1, cmap=my_cmap)
# Adding labels
ax.set_xlabel('Episodes')
ax.set_ylabel('Steps')
ax.set_zlabel('Weights (W1)')
ax.set_title('Weights-1 Evolving Over Episodes')

# plotting actions evolving over episodes for w_2
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_surface(x, y, w_2, cmap=my_cmap)
# Adding labels
ax.set_xlabel('Episodes')
ax.set_ylabel('Steps')
ax.set_zlabel('Weights (W2)')
ax.set_title('Weights-2 Evolving Over Episodes')

# plotting rewards
fig = plt.figure()
plt.plot(rewardarr[0, 0:max_steps], color="blue")
plt.plot(rewardarr[total_episodes // 2, 0:max_steps], color="green")
plt.plot(rewardarr[total_episodes - 1, 0:max_steps], color="brown")
plt.plot(maxreward, color="red")
plt.legend(loc=4)
plt.legend(['Initial Episode', 'Intermediate Episode', 'Last Episode', 'Reward of Optimal Policy'])
plt.title('Rewards In Each Steps')  # Title of the plot
plt.xlabel('Steps')  # X-Label
plt.ylabel('Rewards')  # Y-Label

fig = plt.figure()
plt.plot(rewardavg, color="blue")
plt.title('Sum of Reward')  # Title of the plot
plt.xlabel('Episode')  # X-Label
plt.ylabel('Sum of Reward')  # Y-Label

# plotting error
fig = plt.figure()
plt.plot(erroravg, color="blue")
plt.title('Sum of Error')  # Title of the plot
plt.xlabel('Episode')  # X-Label
plt.ylabel('Sum of Error')  # Y-Label

fig = plt.figure()
plt.subplot(121)
plt.plot(errorarr[0, 0:max_steps], color="red")
plt.plot(minerrorarr[0:max_steps], color="blue")
plt.plot(minerrorarr_1[0:max_steps], color="pink")
plt.plot(error_w_rl[0:max_steps], color="brown")
plt.legend(loc=4)
plt.legend(['First Episode Error', 'Error of Optimal Policy', 'Error of Optimal Policy 1', 'Error without RL'])
plt.title('Error Plot')  # Title of the plot
plt.xlabel('Steps')  # X-Label
plt.ylabel('Error')  # Y-Label
plt.subplot(122)
plt.bar(x_axis_bar, y_axis_bar)

fig = plt.figure()
plt.subplot(421)
plt.plot(UAV1_WRL_Err[0:max_steps], color="red")
plt.plot(UAV1_RL_Err[0:max_steps], color="blue")
plt.legend(['Without RL', 'With RL'])
plt.title('Error Plot Drone')  # Title of the plot
plt.xlabel('Steps')  # X-Label
plt.ylabel('Error')  # Y-Label
plt.subplot(422)
plt.bar(x_ax_uav1, y_ax_uav1)
plt.subplot(423)
plt.plot(UAV2_WRL_Err[0:max_steps], color="red")
plt.plot(UAV2_RL_Err[0:max_steps], color="blue")
plt.legend(['Without RL', 'With RL'])
plt.title('Error Plot FW-UAV 1')  # Title of the plot
plt.xlabel('Steps')  # X-Label
plt.ylabel('Error')  # Y-Label
plt.subplot(424)
plt.bar(x_ax_uav2, y_ax_uav2)
plt.subplot(425)
plt.plot(UAV2_WRL_Err[0:max_steps], color="red")
plt.plot(UAV2_RL_Err[0:max_steps], color="blue")
plt.legend(['Without RL', 'With RL'])
plt.title('Error Plot FW-UAV 2')  # Title of the plot
plt.xlabel('Steps')  # X-Label
plt.ylabel('Error')  # Y-Label
plt.subplot(426)
plt.bar(x_ax_uav3, y_ax_uav3)
plt.subplot(427)
plt.plot(error_w_rl[0:max_steps], color="red")
plt.plot(minerrorarr[0:max_steps], color="blue")
plt.legend(['Without RL', 'With RL'])
plt.title('Total Error Plot')  # Title of the plot
plt.xlabel('Steps')  # X-Label
plt.ylabel('Error')  # Y-Label
plt.subplot(428)
plt.bar(x_ax_uav_f, y_ax_uav_f)

# plottting UAV and target with and without RL
UAV_RL = np.array(UAV_RL)
UAV_W_RL = np.array(UAV_W_RL)
targetarr = np.array(targetarr)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(UAV_RL[0, 0:max_steps], UAV_RL[1, 0:max_steps], UAV_RL[2, 0:max_steps], linewidth="2", color="brown")
ax.plot3D(UAV_W_RL[0, 0:max_steps], UAV_W_RL[1, 0:max_steps], UAV_W_RL[2, 0:max_steps], linewidth="2", color="blue")
ax.plot3D(targetarr[0, 0:max_steps], targetarr[1, 0:max_steps], ss1[0:max_steps], '--', linewidth="2", color="red")
ax.plot3D(UAV_RL[0, 0:max_steps], UAV_RL[1, 0:max_steps], ss1[0:max_steps], linewidth="2", color="pink")
ax.plot3D(UAV_W_RL[0, 0:max_steps], UAV_W_RL[1, 0:max_steps], ss1[0:max_steps], linewidth="2", color="green")
ax.plot3D(x_e_rl[0:100, 0], y_e_rl[0:100, 0], ss1[0:100], linewidth="2", color="black")
ax.plot3D(x_e_w_rl[0:100, 0], y_e_w_rl[0:100, 0], ss1[0:100], linewidth="2", color="orange")
# ax.plot3D(x_e_2[0:100, 0], y_e_2[0:100, 0] , ss1[0:100], linewidth = "2", color = "black")
# ax.plot_surface(Xc_1, Yc_1, Zc_1, cstride = 1,rstride= 1)  # 0bstacles
# ax.plot_surface(Xc_2, Yc_2, Zc_2)
# ax.plot_surface(Xc_3, Yc_3, Zc_3)
ax.set_title('UAV trajectories with and without RL')
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')
ax.legend(['UAV With RL', 'UAV Without RL', 'Target', 'UAV With RL Projection', 'UAV Without RL Projection', 'RL FOV',
           'Without RL FOV'])

fig = plt.figure()
plt.subplot(211)
plt.plot(dist[0:max_steps], linewidth="2", color="blue")
plt.plot(dist2[0:max_steps], linewidth="2", color="green")
plt.plot(dist3[0:max_steps], linewidth="2", color="red")
plt.legend(['Drone - FW-UAV-1', 'FW-UAV-1 - FW-UAV-2', 'FW-UAV-2 - Drone'])
plt.title("Distance b/w UAVs With RL")
plt.xlabel("Iteration")
plt.ylabel("Distance b/w UAVs")
plt.subplot(212)
plt.plot(dist4[0:max_steps], linewidth="2", color="blue")
plt.plot(dist5[0:max_steps], linewidth="2", color="green")
plt.plot(dist6[0:max_steps], linewidth="2", color="red")
plt.legend(['Drone - FW-UAV-1', 'FW-UAV-1 - FW-UAV-2', 'FW-UAV-2 - Drone'])
plt.title("Distance b/w UAVs Without RL")
plt.xlabel("Iteration")
plt.ylabel("Distance b/w UAVs")

# 3D figure plotting by using mayavi final plotting of simulation
fig1 = mlab.figure()
mlab.clf()  # Clear the figure
mlab.plot3d(UAV_RL[0, 0:max_steps], UAV_RL[1, 0:max_steps], UAV_RL[2, 0:max_steps], tube_radius=5,
            color=(255 / 255, 166 / 255, 0))
mlab.plot3d(UAV_W_RL[0, 0:max_steps], UAV_W_RL[1, 0:max_steps], UAV_W_RL[2, 0:max_steps], tube_radius=5,
            color=(0, 1, 0))
mlab.plot3d(UAV_RL[0 + 8, 0:max_steps], UAV_RL[1 + 8, 0:max_steps], UAV_RL[2 + 8, 0:max_steps], tube_radius=5,
            color=(255 / 255, 166 / 255, 0))
mlab.plot3d(UAV_W_RL[0 + 8, 0:max_steps], UAV_W_RL[1 + 8, 0:max_steps], UAV_W_RL[2 + 8, 0:max_steps], tube_radius=5,
            color=(0, 1, 0))
mlab.plot3d(UAV_RL[0, 0:max_steps], UAV_RL[1, 0:max_steps], ss1[0:max_steps], tube_radius=4,
            color=(255 / 255, 166 / 255, 0))
mlab.plot3d(UAV_W_RL[0, 0:max_steps], UAV_W_RL[1, 0:max_steps], ss1[0:max_steps], tube_radius=4, color=(0, 1, 0))
mlab.plot3d(UAV_RL[0 + 8, 0:max_steps], UAV_RL[1 + 8, 0:max_steps], ss1[0:max_steps], tube_radius=4,
            color=(255 / 255, 166 / 255, 0))
mlab.plot3d(UAV_W_RL[0 + 8, 0:max_steps], UAV_W_RL[1 + 8, 0:max_steps], ss1[0:max_steps], tube_radius=4,
            color=(0, 1, 0))
mlab.plot3d(UAV1_cen[0:max_steps, 0], UAV1_cen[0:max_steps, 1], ss1[0:max_steps], tube_radius=2,
            color=(0, 0, 0))
mlab.plot3d(UAV2_cen[0:max_steps, 0], UAV2_cen[0:max_steps, 1], ss1[0:max_steps], tube_radius=2,
            color=(0, 0, 0))

mlab.plot3d(targetarr[0, 0:max_steps], targetarr[1, 0:max_steps], ss1[0:max_steps], tube_radius=5, color=(0, .7, 0))
mlab.plot3d(targetarr[0 + 3, 0:max_steps], targetarr[1 + 3, 0:max_steps], ss1[0:max_steps], tube_radius=5,
            color=(0, .7, 0))
mlab.plot3d(targetarr[0 + 3 + 3, 0:max_steps], targetarr[1 + 3 + 3, 0:max_steps], ss1[0:max_steps], tube_radius=5,
            color=(0, .7, 0))

mlab.plot3d(x_e_rl[0:100, 0], y_e_rl[0:100, 0], ss1[0:100], tube_radius=3, color=(0, 0, 0))
mlab.plot3d(x_e_w_rl[0:100, 0], y_e_w_rl[0:100, 0], ss1[0:100], tube_radius=3, color=(1, 1, 1))
mlab.plot3d(x_e_rl_1[0:100, 0], y_e_rl_1[0:100, 0], ss1[0:100], tube_radius=3, color=(0, 0, 0))
mlab.plot3d(x_e_w_rl_1[0:100, 0], y_e_w_rl_1[0:100, 0], ss1[0:100], tube_radius=3, color=(1, 1, 1))

mlab.mesh(Xc_1, Yc_1, Zc_1)
mlab.mesh(Xc_2, Yc_2, Zc_2)
mlab.mesh(Xc_3, Yc_3, Zc_3)
mlab.mesh(Xc_4, Yc_4, Zc_4)
mlab.mesh(Xc_5, Yc_5, Zc_5)
mlab.mesh(Xc_6, Yc_6, Zc_6)
mlab.mesh(Xc_7, Yc_7, Zc_7)
mlab.mesh(Xc_8, Yc_8, Zc_8)
mlab.mesh(Xc_9, Yc_9, Zc_9)
mlab.mesh(Xc_10, Yc_10, Zc_10)

mlab.title('Tracking UAV')
s = [0, 1900, -300, 1000, 0, 170]
mlab.orientation_axes(xlabel='X-axis', ylabel='Y-axis', zlabel='Z-axis')
mlab.outline(color=(0, 0, 0), extent=s)
mlab.axes(color=(0, 0, 0), extent=s)
plt.show()
mlab.show()
