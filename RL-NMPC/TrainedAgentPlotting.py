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
from tvtk.api import tvtk
from mayavi.modules.api import Outline, GridPlane

gym.logger.set_level(40)

"""
This code is for tunning Model Predictive Controller with 
Reinforcement learning specifically by using Q-learning algorithm.
Custom environment created by OpenAI gym and CasADi used framework
for solving constrained optimization problem of MPC.
Description:
    Model Predictive controller is controlling UAV to follow target.
    Goal with the reinforcement learning is to tune MPC weights for 
    getting optimal performace. 
Source:
    Will add details about publication
Observation:
    Type: Box(3)
        Num     Observation               Min                     Max
        0       X position of UAV         -Inf                    Inf
        1       Y position of UAV         -Inf                    Inf
        2       Z position of UAV         -Inf                    Inf  
Action:
    Type: Tuple(Discrete(11),Discrete(11))
Reward:
    Reward is based of error, error is euclidean distance between 
    center of FOV to targets location.
    if Error --> INF then Reward --> 0
       Error --> 0 then Reward --> INF
Initial State:
    UAV starts from the initial position
    x0 = [90, 150, 80, 0, 0, 0, 0, 0]
Episode termination:
    Episode terminates after fixed steps.
"""

#########################################################################
##################### MPC and Supportive functions ######################

# Global MPC variables
x0 = [99, 150, 200, 0, 0, 0, 0, 0, 99, 200, 200, 0, 0, 0, 0, 0, 99, 250, 200, 0, 0, 0, 0, 0, 0, 0, 0]
xs = [100, 150, pi / 4, 100, 200, pi / 4, 100, 250, pi / 4]
max_step_size = 1426
mpc_iter = 0
mpc = 0
sc = 0
UAV1_FOV_Plot_RL = ca.DM.zeros((max_step_size + 1, 4))
UAV2_FOV_Plot_RL = ca.DM.zeros((max_step_size + 1, 4))
UAV3_FOV_Plot_RL = ca.DM.zeros((max_step_size + 1, 4))
ss1 = np.zeros((max_step_size + 1000))
controls_s = ca.DM.zeros((6+6+6, max_step_size))
controls_o = ca.DM.zeros((6+6+6, max_step_size))

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
P = ca.SX.sym('P', n_states_u + 9 + 6)  # This consists of initial states of UAV with gimbal 1-8 and
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
          P[36] * ca.sqrt((stt[0, k] - P[27]) ** 2 + (stt[1, k] - P[28]) ** 2) + \
          P[37] * ((A[k] * (P[27] - X_E[k]) ** 2 + B[k] * (P[28] - Y_E[k]) * (P[27] - X_E[k])
                 + C[k] * (P[28] - Y_E[k]) ** 2) - 1) + \
          P[38] * ca.sqrt((stt[8, k] - P[30]) ** 2 + (stt[9, k] - P[31]) ** 2) + \
          P[39] * ((A_1[k] * (P[30] - X_E_1[k]) ** 2 + B_1[k] * (P[31] - Y_E_1[k]) * (P[30] - X_E_1[k])
                 + C_1[k] * (P[31] - Y_E_1[k]) ** 2) - 1) + \
          P[40] * ca.sqrt((stt[16, k] - P[33]) ** 2 + (stt[17, k] - P[34]) ** 2) + \
          P[41] * ((A_2[k] * (P[33] - X_E_2[k]) ** 2 + B_2[k] * (P[34] - Y_E_2[k]) * (P[33] - X_E_2[k])
                 + C_2[k] * (P[34] - Y_E_2[k]) ** 2) - 1)


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
                   X[26, k],
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
        'max_iter': 100,
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
    if (sc >= 500):
        con_t = [v, 0, v_1, 0, v_2, 0]  # left ahead
    if (sc >= 700 + 10 + 1 + 1):
        con_t = [v, -3 * pi / 200, v_1, -3 * pi / 200, v_2, -3 * pi / 200]  # 50 ahead
    if (sc >= 1200 + 10 + 1 + 1):
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


# convert DM and SX to array
def DM2Arr(dm):
    return np.array(dm.full())


def SX2Arr(sx):
    return np.array(sx.full())


# For plotting ellipse
def ellipse(a_p, b_p, x_e_1, y_e_1):
    x = ca.DM.zeros((101))
    y = ca.DM.zeros((101))
    th = np.linspace(0, 2 * np.pi, 100)
    x = a_p * sin(th) + x_e_1
    y = b_p * cos(th) + y_e_1
    return x, y


def max_index(arr):
    shape = np.shape(arr)
    max_indices = (0, 0)
    for i in range(shape[0]):
        for k in range(shape[1]):
            if arr[max_indices] < arr[i, k]:
                max_indices = (i, k)
    return max_indices


# main MPC function
def MPC(w1, w2, w3, w4, w5, w6):
    global T, N, v_u_min, v_u_max, omega_2_u_min, omega_2_u_max, omega_3_u_min , omega_3_u_max, omega_1_g_min, \
        omega_1_g_max, omega_2_g_min, omega_2_g_max, omega_3_g_min, omega_3_g_max, theta_u_min, theta_u_max, z_u_min, \
        z_u_max, phi_g_min, phi_g_max, theta_g_min, theta_g_max, shi_g_min, shi_g_max, x_o_1, y_o_1, obs_r, x_o_2, \
        y_o_2, x_o_3, y_o_3, x_u, y_u, z_u, theta_u, psi_u, phi_g, shi_g, theta_g, states_u, n_states_u, v_u, omega_2_u,\
        omega_3_u, omega_1_g, omega_2_g, omega_3_g, controls_u, n_controls, rhs_u, f_u, U, P, X, lbg, ubg, lbx, ubx, ff,\
        obj, g, OPT_variables, nlp_prob, opts, solver, args, x0, xs, mpc_iter, sc, \
        v_u_min_1, v_u_max_1, omega_2_u_min_1, omega_2_u_max_1, omega_3_u_min_1, omega_3_u_max_1, omega_1_g_min_1, \
        omega_1_g_max_1, omega_2_g_min_1, omega_2_g_max_1, omega_3_g_min_1, omega_3_g_max_1, theta_u_min_1, \
        theta_u_max_1, z_u_min_1, z_u_max_1, phi_g_min_1, phi_g_max_1, theta_g_min_1, theta_g_max_1, shi_g_min_1, \
        shi_g_max_1, x_u_1, y_u_1, z_u_1, theta_u_1, psi_u_1, phi_g_1, shi_g_1, theta_g_1, v_u_1, omega_2_u_1, \
        omega_3_u_1, omega_1_g_1, omega_2_g_1, omega_3_g_1, \
        v_u_min_2, v_u_max_2, omega_2_u_min_2, omega_2_u_max_2, omega_3_u_min_2, omega_3_u_max_2, omega_1_g_min_2, \
        omega_1_g_max_2, omega_2_g_min_2, omega_2_g_max_2, omega_3_g_min_2, omega_3_g_max_2, theta_u_min_2, \
        theta_u_max_2, z_u_min_2, z_u_max_2, phi_g_min_2, phi_g_max_2, theta_g_min_2, \
        theta_g_max_2, shi_g_min_2, shi_g_max_2, x_u_2, y_u_2, z_u_2, theta_u_2, psi_u_2, phi_g_2, shi_g_2, theta_g_2,\
        v_u_2, omega_2_u_2, omega_3_u_2, omega_1_g_2, omega_2_g_2, omega_3_g_2, rel_f_u, mpc

    t1 = time()
    t0 = 0

    # xx = DM(state_init)
    t = ca.DM(t0)

    u0 = ca.DM.zeros((n_controls, N))
    xx = ca.DM.zeros((27, max_step_size+1))
    # ss = ca.DM.zeros((3, 801))

    x_e_1 = ca.DM.zeros((max_step_size+1))
    y_e_1 = ca.DM.zeros((max_step_size+1))
    x_e_5 = ca.DM.zeros((max_step_size+1))
    y_e_5 = ca.DM.zeros((max_step_size+1))
    x_e_6 = ca.DM.zeros((max_step_size+1))
    y_e_6 = ca.DM.zeros((max_step_size+1))

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
        w1,
        w2,
        w3,
        w4,
        w5,
        w6   # target state
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
    # print(mpc)
    if (mpc < max_step_size):
        # print(u[:, 0])
        controls_s[:, sc] = u[:, 0]
        # print(controls_s)
    controls_o[:, sc] = u[:, 0]

    # xx ...
    t2 = time()
    # print(t2-t1)
    times = np.vstack((
        times,
        t2 - t1
    ))

    sc = sc + 1
    mpc_iter = mpc_iter + 1
    mpc = mpc + 1

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

    UAV1_FOV_Plot_RL[sc, 0] = a_p
    UAV1_FOV_Plot_RL[sc, 1] = b_p
    UAV1_FOV_Plot_RL[sc, 2] = x_e_1[mpc_iter]
    UAV1_FOV_Plot_RL[sc, 3] = y_e_1[mpc_iter]

    UAV2_FOV_Plot_RL[sc, 0] = a_p_1
    UAV2_FOV_Plot_RL[sc, 1] = b_p_1
    UAV2_FOV_Plot_RL[sc, 2] = x_e_5[mpc_iter]
    UAV2_FOV_Plot_RL[sc, 3] = y_e_5[mpc_iter]

    UAV3_FOV_Plot_RL[sc, 0] = a_p_2
    UAV3_FOV_Plot_RL[sc, 1] = b_p_2
    UAV3_FOV_Plot_RL[sc, 2] = x_e_6[mpc_iter]
    UAV3_FOV_Plot_RL[sc, 3] = y_e_6[mpc_iter]

    Error0 = ca.sqrt((x_e_1[mpc_iter] - ss[0, mpc_iter-1]) ** 2 + (y_e_1[mpc_iter] - ss[1, mpc_iter-1]) ** 2)
    Error1 = ca.sqrt((x_e_5[mpc_iter] - ss[3, mpc_iter - 1]) ** 2 + (y_e_5[mpc_iter] - ss[4, mpc_iter - 1]) ** 2)
    Error2 = ca.sqrt((x_e_6[mpc_iter] - ss[6, mpc_iter - 1]) ** 2 + (y_e_6[mpc_iter] - ss[7, mpc_iter - 1]) ** 2)
    Error3 = Error0 + Error1 + Error2

    return Error0, Error1, Error2, Error3, x0[0:27], xs, a_p, b_p, x_e_1[mpc_iter], y_e_1[mpc_iter], a_p_1, b_p_1, \
                 x_e_5[mpc_iter], y_e_5[mpc_iter], a_p_2, b_p_2, x_e_6[mpc_iter], y_e_6[mpc_iter]



#################################################################################
############# Defining "tunning" reinforcement learning environment #############

class Tunning(Env):
    def __init__(self):
        # Action space which contains two discrete actions which are weights of MPC
        self.action_space = Tuple(spaces=(Discrete(101), Discrete(101), Discrete(101), Discrete(101), Discrete(101),
                                          Discrete(101)))
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
        error, error1, error2, error3, obs, obs2, a_p, b_p, x_e, y_e, a_p_1, b_p_1, x_e_1, y_e_1, a_p_2, b_p_2, x_e_2, \
        y_e_2 = MPC(action[0], action[1], action[2], action[3], action[4], action[5])

        # reward = 1/(error + error3)

        reward = 1/error
        reward1 = 1/error1
        reward2 = 1/error2

        # Check if episode is done or not
        if self.episode_length <= 0:
            done = True
        else:
            done = False

        # extra info about env
        info = {}

        # Return step information
        return obs, obs2, reward, reward1, reward2, done, info, error, error1, error2, error3, a_p, b_p, x_e, y_e, \
               a_p_1, b_p_1, x_e_1, y_e_1, a_p_2, b_p_2, x_e_2, y_e_2

    def render(self):
        # Implement viz
        pass

    def reset(self):
        # Reset UAV & target to initial position
        global x0, xs, mpc_iter, max_step_size, ss, sc
        x0 = [99, 150, 200, 0, 0, 0, 0, 0, 99, 200, 200, 0, 0, 0, 0, 0, 99, 250, 200, 0, 0, 0, 0, 0, 0, 0, 0]
        xs = [100, 150, pi/4, 100, 200, pi/4, 100, 250, pi/4]
        sc = 0
        mpc_iter = 0
        ss = ca.DM.zeros((3+3+3, max_step_size))
        self.episode_length = max_step_size


env = Tunning()
#################################################################
########################### Q-learning ##########################
step_size = max_step_size + 1  # Change according to main loop run

# Q - learning parameters
total_episodes = 150  # Total episodes
max_steps = max_step_size  # Max steps per episode

step_size = max_step_size + 1  # Change according to main loop run

# List and array of rewards, errors etc.
rewards = []
rewardarr = np.zeros((total_episodes, max_steps))
errorarr = np.zeros((total_episodes, max_steps))
erroravg = np.zeros((total_episodes))
rewardavg = np.zeros((total_episodes))
maxreward = np.zeros((max_steps))
UAV1_RL_Err = np.zeros((max_steps))
UAV2_RL_Err = np.zeros((max_steps))
UAV3_RL_Err = np.zeros((max_steps))
UAV1_WRL_Err = np.zeros((max_steps))
UAV2_WRL_Err = np.zeros((max_steps))
UAV3_WRL_Err = np.zeros((max_steps))
minerrorarr = np.zeros((max_steps))
action_str = np.zeros((max_steps + 1, 2))
w_1 = np.zeros((max_steps, total_episodes))
w_2 = np.zeros((max_steps, total_episodes))
w_3 = np.zeros((max_steps, total_episodes))
w_4 = np.zeros((max_steps, total_episodes))
UAV_RL = ca.DM.zeros((27, max_step_size + 1))
UAV_W_RL = ca.DM.zeros((27, max_step_size + 1))
targetarr = ca.DM.zeros((9, max_step_size + 1))
error_w_rl = np.zeros((max_steps + 1))
FOV_C_RL = ca.DM.zeros((max_step_size, 2))
FOV_C_W_RL = ca.DM.zeros((max_step_size, 2))
FOV_C_RL_1 = ca.DM.zeros((max_step_size, 2))
FOV_C_W_RL_1 = ca.DM.zeros((max_step_size, 2))
FOV_C_RL_2 = ca.DM.zeros((max_step_size, 2))
FOV_C_W_RL_2 = ca.DM.zeros((max_step_size, 2))

w1 = np.zeros((max_step_size))
w2 = np.zeros((max_step_size))
w3 = np.zeros((max_step_size))
w4 = np.zeros((max_step_size))
w5 = np.zeros((max_step_size))
w6 = np.zeros((max_step_size))
times = ca.DM.zeros((max_step_size))

##
UAV1_FOVP = np.zeros((max_steps, 2))
UAV2_FOVP = np.zeros((max_steps, 2))
UAV3_FOVP = np.zeros((max_steps, 2))

UAV1_W_FOVP = np.zeros((max_steps, 2))
UAV2_W_FOVP = np.zeros((max_steps, 2))
UAV3_W_FOVP = np.zeros((max_steps, 2))

##
linear_vel_min = np.zeros((max_step_size))
linear_vel_max = np.zeros((max_step_size))

linear_acc_min = np.zeros((max_step_size))
linear_acc_max = np.zeros((max_step_size))

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

linear_acc_min_1 = np.zeros((max_step_size))
linear_acc_max_1 = np.zeros((max_step_size))

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

linear_acc_min_2 = np.zeros((max_step_size))
linear_acc_max_2 = np.zeros((max_step_size))

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
arr = np.load("Jo_test_150ep_final.npz")
arr2 = np.load("Jo_WRL_Controls_final.npz")
error = arr2["error"]
error_1 = arr2["error_1"]
error_2 = arr2["error_2"]
control_effort_WRL1 = arr2["control_effort1"]
control_effort_WRL2 = arr2["control_effort2"]
control_effort_WRL3 = arr2["control_effort3"]
control_effort_WRL4 = arr2["control_effort4"]
control_effort_WRL5 = arr2["control_effort5"]
control_effort_WRL6 = arr2["control_effort6"]
control_effort_WRL7 = arr2["control_effort7"]
control_effort_WRL8 = arr2["control_effort8"]
control_effort_WRL9 = arr2["control_effort9"]
control_effort_WRL10 = arr2["control_effort10"]
control_effort_WRL11 = arr2["control_effort11"]
control_effort_WRL12 = arr2["control_effort12"]
control_effort_WRL13 = arr2["control_effort13"]
control_effort_WRL14 = arr2["control_effort14"]
control_effort_WRL15 = arr2["control_effort15"]
control_effort_WRL16 = arr2["control_effort16"]
control_effort_WRL17 = arr2["control_effort17"]
control_effort_WRL18 = arr2["control_effort18"]
errorarr = arr['errorarr']
total_episodes = arr['total_episodes']
w_1 = arr['w_1']
w_2 = arr['w_2']
w_3 = arr['w_3']
w_4 = arr['w_4']
w_5 = arr['w_5']
w_6 = arr['w_6']
rewardarr = arr['rewardarr']
erroravg = arr['erroravg']
rewardavg = arr['rewardavg']
qtable = arr['qtable']
qtable1 = arr['qtable1']
qtable2 = arr['qtable2']


error1_uni = np.array(error)
error2_uni = np.array(error_1)
error3_uni = np.array(error_2)

sum_err = sum(error1_uni[0:max_step_size])
sum_err_1 = sum(error2_uni[0:max_step_size])
sum_err_2 = sum(error3_uni[0:max_step_size])
final_error = sum_err + sum_err_1 + sum_err_2

final_err_arr = np.zeros((max_step_size))
for i in range(max_step_size):
    final_err_arr[i] = error1_uni[i] + error2_uni[i] + error3_uni[i]

# Collecing data of obstacles for plotting cylinders
Xc_1, Yc_1, Zc_1 = data_for_cylinder_along_z(x_o_1, y_o_1, obs_r, 250)
Xc_2, Yc_2, Zc_2 = data_for_cylinder_along_z(x_o_2, y_o_2, obs_r, 250)
Xc_3, Yc_3, Zc_3 = data_for_cylinder_along_z(x_o_3, y_o_3, obs_r, 250)
Xc_4, Yc_4, Zc_4 = data_for_cylinder_along_z(x_o_4, y_o_4, obs_r, 250)
Xc_5, Yc_5, Zc_5 = data_for_cylinder_along_z(x_o_5, y_o_5, obs_r, 250)
Xc_6, Yc_6, Zc_6 = data_for_cylinder_along_z(x_o_6, y_o_6, obs_r, 250)
Xc_7, Yc_7, Zc_7 = data_for_cylinder_along_z(x_o_7, y_o_7, obs_r, 250)
Xc_8, Yc_8, Zc_8 = data_for_cylinder_along_z(x_o_8, y_o_8, obs_r, 250)
# Xc_9, Yc_9, Zc_9 = data_for_cylinder_along_z(x_o_9, y_o_9, obs_r, 250)
# Xc_10, Yc_10, Zc_10 = data_for_cylinder_along_z(x_o_10, y_o_10, obs_r, 250)

for i in range(max_step_size):
    times[i] = i * 0.2

times = np.array(times)
times1 = np.squeeze(times)
print(times1.shape)
print(times.shape)

times = np.reshape(times, (1, max_step_size))
print(times.shape)

for i in range(max_step_size):
    linear_vel_min[i] = 0
    linear_vel_max[i] = 30
    linear_acc_min[i] = -10
    linear_acc_max[i] = 10
    pitch_rate_min[i] = -pi / 30
    pitch_rate_max[i] = pi / 30
    yaw_rate_min[i] = -(pi / 21)
    yaw_rate_max[i] =  (pi / 21)
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
    linear_acc_min_1[i] = -10
    linear_acc_max_1[i] = 10
    pitch_rate_min_1[i] = -pi / 30
    pitch_rate_max_1[i] = pi / 30
    yaw_rate_min_1[i] = -7 *(pi / 21)
    yaw_rate_max_1[i] = 7 *(pi / 21)
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
    linear_acc_min_2[i] = -10
    linear_acc_max_2[i] = 10
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

## new

Error = 0
env.reset()
UAV_RL[:, 0] = x0[0:27]
targetarr[:, 0] = xs
# Printing Optimal Policy
for i in range(max_step_size):
    action1 = max_index(qtable[i, :, :])
    action2 = max_index(qtable1[i, :, :])
    action3 = max_index(qtable2[i, :, :])
    action = (action1[0], action1[1], action2[0], action2[1], action3[0], action3[1])
    new_state, obs2, reward, reward1, reward2, done, info, error, error1, error2, error3, a_p, b_p, x_e, y_e, a_p_1, \
    b_p_1, x_e_1, y_e_1, a_p_2, b_p_2, x_e_2, y_e_2 = env.step(action)
    w1[i] = action[0]
    w2[i] = action[1]
    w3[i] = action2[0]
    w4[i] = action2[1]
    w5[i] = action3[0]
    w6[i] = action3[1]
    UAV1_RL_Err[i] = error
    UAV2_RL_Err[i] = error1
    UAV3_RL_Err[i] = error2
    UAV_RL[:, i + 1] = new_state
    # print(new_state)
    FOV_C_RL[i, 0] = x_e
    FOV_C_RL[i, 1] = y_e
    FOV_C_RL_1[i, 0] = x_e_1
    FOV_C_RL_1[i, 1] = y_e_1
    FOV_C_RL_2[i, 0] = x_e_2
    FOV_C_RL_2[i, 1] = y_e_2
    UAV1_FOVP[i, 0] = a_p
    UAV1_FOVP[i, 1] = b_p
    UAV2_FOVP[i, 0] = a_p_1
    UAV2_FOVP[i, 1] = b_p_1
    UAV3_FOVP[i, 0] = a_p_2
    UAV3_FOVP[i, 1] = b_p_2
    targetarr[:, i + 1] = obs2
    maxreward[i] = reward
    minerrorarr[i] = error3
    Error += error3
    print(i)

FOV_C_RL = np.array(FOV_C_RL)
FOV_C_RL_1 = np.array(FOV_C_RL)
FOV_C_RL_2 = np.array(FOV_C_RL)

A1_control1 = controls_s[0, 0:max_steps]
print(controls_s[0, 0:max_steps])
A1_control2 = controls_s[1, 0:max_steps]
A1_control3 = controls_s[2, 0:max_steps]
A1_control4 = controls_s[3, 0:max_steps]
A1_control5 = controls_s[4, 0:max_steps]
A1_control6 = controls_s[5, 0:max_steps]
A2_control1 = controls_s[6, 0:max_steps]
A2_control2 = controls_s[7, 0:max_steps]
A2_control3 = controls_s[8, 0:max_steps]
A2_control4 = controls_s[9, 0:max_steps]
A2_control5 = controls_s[10, 0:max_steps]
A2_control6 = controls_s[11, 0:max_steps]
A3_control1 = controls_s[12, 0:max_steps]
A3_control2 = controls_s[13, 0:max_steps]
A3_control3 = controls_s[14, 0:max_steps]
A3_control4 = controls_s[15, 0:max_steps]
A3_control5 = controls_s[16, 0:max_steps]
A3_control6 = controls_s[17, 0:max_steps]

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
control_squared16= np.square(A3_control4)
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
x_e_rl_2, y_e_rl_2 = ellipse(a_p_2, b_p_2, x_e_2, y_e_2)
x_e_rl_2 = np.array(x_e_rl_2)
y_e_rl_2 = np.array(y_e_rl_2)
print('Error of Optimal Policy: {}'.format(Error))

# for without RL error and trajectory
Error = 0
env.reset()
UAV_W_RL[:, 0] = x0[0:27]
# Printing Optimal Policy
for i in range(max_step_size-1300):
    action = (1, 1, 1, 1, 1, 1)
    # mpc_iter = mpc_iter + 1
    new_state, obs2, reward, reward1, reward2, done, info, error, error1, error2, error3, a_p, b_p, x_e, y_e, a_p_1, \
    b_p_1, x_e_1, y_e_1, a_p_2, b_p_2, x_e_2, y_e_2 = env.step(action)
    UAV1_WRL_Err[i] = error
    UAV2_WRL_Err[i] = error1
    UAV3_WRL_Err[i] = error2
    UAV_W_RL[:, i + 1] = new_state
    error_w_rl[i] = error3
    FOV_C_W_RL[i, 0] = x_e
    FOV_C_W_RL[i, 1] = y_e
    FOV_C_W_RL_1[i, 0] = x_e_1
    FOV_C_W_RL_1[i, 1] = y_e_1
    FOV_C_W_RL_2[i, 0] = x_e_2
    FOV_C_W_RL_2[i, 1] = y_e_2
    UAV1_W_FOVP[i, 0] = a_p
    UAV1_W_FOVP[i, 1] = b_p
    UAV2_W_FOVP[i, 0] = a_p_1
    UAV2_W_FOVP[i, 1] = b_p_1
    UAV3_W_FOVP[i, 0] = a_p_2
    UAV3_W_FOVP[i, 1] = b_p_2
    Error += error3
    print(i)

dist4 = ca.DM.zeros(max_step_size + 1)
for i in range(max_step_size):
    dist4[i] = ca.sqrt((UAV_RL[0, i] - UAV_RL[8, i]) ** 2 + (UAV_RL[1, i] - UAV_RL[9, i]) ** 2)

dist5 = ca.DM.zeros(max_step_size + 1)
for i in range(max_step_size):
    dist5[i] = ca.sqrt((UAV_RL[16, i] - UAV_RL[8, i]) ** 2 + (UAV_RL[17, i] - UAV_RL[9, i]) ** 2)

dist6 = ca.DM.zeros(max_step_size + 1)
for i in range(max_step_size):
    dist6[i] = ca.sqrt((UAV_RL[0, i] - UAV_RL[16, i]) ** 2 + (UAV_RL[1, i] - UAV_RL[17, i]) ** 2)

A1_control1 = controls_o[0, 0:max_steps]
print(controls_o[0, 0:max_steps])
A1_control2 = controls_o[1, 0:max_steps]
A1_control3 = controls_o[2, 0:max_steps]
A1_control4 = controls_o[3, 0:max_steps]
A1_control5 = controls_o[4, 0:max_steps]
A1_control6 = controls_o[5, 0:max_steps]
A2_control1 = controls_o[6, 0:max_steps]
A2_control2 = controls_o[7, 0:max_steps]
A2_control3 = controls_o[8, 0:max_steps]
A2_control4 = controls_o[9, 0:max_steps]
A2_control5 = controls_o[10, 0:max_steps]
A2_control6 = controls_o[11, 0:max_steps]
A3_control1 = controls_o[12, 0:max_steps]
A3_control2 = controls_o[13, 0:max_steps]
A3_control3 = controls_o[14, 0:max_steps]
A3_control4 = controls_o[15, 0:max_steps]
A3_control5 = controls_o[16, 0:max_steps]
A3_control6 = controls_o[17, 0:max_steps]

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
control_squared16= np.square(A3_control4)
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

control_effort_WRL1 = sqroot1/285.2
print(control_effort_WRL1)
control_effort_WRL2 = sqroot2/285.2
control_effort_WRL3 = sqroot3/285.2
control_effort_WRL4 = sqroot4/285.2
control_effort_WRL5 = sqroot5/285.2
control_effort_WRL6 = sqroot6/285.2
control_effort_WRL7 = sqroot7/285.2
control_effort_WRL8 = sqroot8/285.2
control_effort_WRL9 = sqroot9/285.2
control_effort_WRL10 = sqroot10/285.2
control_effort_WRL11 = sqroot11/285.2
control_effort_WRL12 = sqroot12/285.2
control_effort_WRL13 = sqroot13/285.2
control_effort_WRL14 = sqroot14/285.2
control_effort_WRL15 = sqroot15/285.2
control_effort_WRL16 = sqroot16/285.2
control_effort_WRL17 = sqroot17/285.2
control_effort_WRL18 = sqroot18/285.2


dist4 = np.array(dist4)
dist5 = np.array(dist5)
dist6 = np.array(dist6)

x_e_w_rl, y_e_w_rl = ellipse(a_p, b_p, x_e, y_e)
x_e_w_rl = np.array(x_e_w_rl)
y_e_w_rl = np.array(y_e_w_rl)
x_e_w_rl_1, y_e_w_rl_1 = ellipse(a_p_1, b_p_1, x_e_1, y_e_1)
x_e_w_rl_1 = np.array(x_e_w_rl_1)
y_e_w_rl_1 = np.array(y_e_w_rl_1)
x_e_w_rl_2, y_e_w_rl_2 = ellipse(a_p_2, b_p_2, x_e_2, y_e_2)
x_e_w_rl_2 = np.array(x_e_w_rl_2)
y_e_w_rl_2 = np.array(y_e_w_rl_2)
print('Error without RL: {}'.format(Error))

################################################################
############# Plotting the reward & printing actions ###########
controls_s = np.array(controls_s)
controls_o = np.array(controls_o)
sum_without_rl = sum(error_w_rl)
sum_first_episode = sum(errorarr[0, 0:max_steps])
sum_last_episode = sum(errorarr[total_episodes-1, 0:max_steps])
sum_optimal_policy = sum(minerrorarr)

# New
sum_error_uav1_rl = sum(UAV1_RL_Err)
sum_error_uav2_rl = sum(UAV2_RL_Err)
sum_error_uav3_rl = sum(UAV3_RL_Err)
sum_error_uav1_wrl = sum(UAV1_WRL_Err)
sum_error_uav2_wrl = sum(UAV2_WRL_Err)
sum_error_uav3_wrl = sum(UAV3_WRL_Err)

x_ax_uav1 = ['Error without RL', 'Error of OptimalPolicy']
y_ax_uav1 = [sum_err, sum_error_uav1_rl]

x_ax_uav2 = ['Error without RL', 'Error of OptimalPolicy']
y_ax_uav2 = [sum_err_1, sum_error_uav2_rl]

x_ax_uav3 = ['Error without RL', 'Error of OptimalPolicy']
y_ax_uav3 = [sum_err_2, sum_error_uav3_rl]

x_ax_uav_f = ['Total Error without RL', 'Total Error of OptimalPolicy']
y_ax_uav_f = [final_error, sum_optimal_policy]

x_control_effort1 = ['Acceleration Effort of Untrained Agent', 'Acceleration Effort of RL Trained Agent']
y_control_effort1 = [control_effort_WRL1, control_effort1]

x_control_effort2 = ['Pitch Effort of Untrained Agent', 'Pitch Effort of RL Trained Agent']
y_control_effort2 = [control_effort_WRL2, control_effort2]

x_control_effort3 = ['Yaw Effort of Untrained Agent', 'Yaw Effort of RL Trained Agent']
y_control_effort3 = [control_effort_WRL3, control_effort3]

x_control_effort4 = ['Gimbal Roll Effort of Untrained Agent', 'Gimbal Roll Effort of RL Trained Agent']
y_control_effort4 = [control_effort_WRL4, control_effort4]

x_control_effort5 = ['Gimbal Pitch Effort of Untrained Agent', 'Gimbal Pitch Effort of RL Trained Agent']
y_control_effort5 = [control_effort_WRL5, control_effort5]

x_control_effort6 = ['Gimbal Yaw Effort of Untrained Agent', 'Gimbal Roll Yaw of RL Trained Agent']
y_control_effort6 = [control_effort_WRL6, control_effort6]

x_control_effort7 = ['Acceleration Effort of Untrained Agent', 'Acceleration Effort of RL Trained Agent']
y_control_effort7 = [control_effort_WRL7, control_effort7]

x_control_effort8 = ['Pitch Effort of Untrained Agent', 'Pitch Effort of RL Trained Agent']
y_control_effort8 = [control_effort_WRL8, control_effort8]

x_control_effort9 = ['Yaw Effort of Untrained Agent', 'Yaw Effort of RL Trained Agent']
y_control_effort9 = [control_effort_WRL9, control_effort9]

x_control_effort10 = ['Gimbal Roll Effort of Untrained Agent', 'Gimbal Roll Effort of RL Trained Agent']
y_control_effort10 = [control_effort_WRL10, control_effort10]

x_control_effort11 = ['Gimbal Pitch Effort of Untrained Agent', 'Gimbal Pitch Effort of RL Trained Agent']
y_control_effort11 = [control_effort_WRL11, control_effort11]

x_control_effort12 = ['Gimbal Yaw Effort of Untrained Agent', 'Gimbal Roll Yaw of RL Trained Agent']
y_control_effort12 = [control_effort_WRL12, control_effort12]

x_control_effort13 = ['Acceleration Effort of Untrained Agent', 'Acceleration Effort of RL Trained Agent']
y_control_effort13 = [control_effort_WRL13, control_effort13]

x_control_effort14 = ['Pitch Effort of Untrained Agent', 'Pitch Effort of RL Trained Agent']
y_control_effort14 = [control_effort_WRL14, control_effort14]

x_control_effort15 = ['Yaw Effort of Untrained Agent', 'Yaw Effort of RL Trained Agent']
y_control_effort15 = [control_effort_WRL15, control_effort15]

x_control_effort16 = ['Gimbal Roll Effort of Untrained Agent', 'Gimbal Roll Effort of RL Trained Agent']
y_control_effort16 = [control_effort_WRL16, control_effort16]

x_control_effort17 = ['Gimbal Pitch Effort of Untrained Agent', 'Gimbal Pitch Effort of RL Trained Agent']
y_control_effort17 = [control_effort_WRL17, control_effort17]

x_control_effort18 = ['Gimbal Yaw Effort of Untrained Agent', 'Gimbal Roll Yaw Effort of RL Trained Agent']
y_control_effort18 = [control_effort_WRL18, control_effort18]

# Printing actions
my_cmap = plt.get_cmap('cool')

UAV_RL = np.array(UAV_RL)
FOV_C_RL = np.array(FOV_C_RL)
FOV_C_RL_1 = np.array(FOV_C_RL_1)
FOV_C_RL_2 = np.array(FOV_C_RL_2)
UAV1_FOVP = np.array(UAV1_FOVP)
UAV2_FOVP = np.array(UAV2_FOVP)
UAV3_FOVP = np.array(UAV3_FOVP)

np.savez("Jo_RL_Plot3D_final.npz", UAV_RL=UAV_RL, UAV_W_RL=UAV_W_RL, FOV_C_RL=FOV_C_RL, FOV_C_RL_1=FOV_C_RL_1,
         FOV_C_RL_2=FOV_C_RL_2, FOV_C_W_RL=FOV_C_W_RL, FOV_C_W_RL_1=FOV_C_W_RL_1, FOV_C_W_RL_2=FOV_C_W_RL_2,
         UAV1_FOVP=UAV1_FOVP, UAV2_FOVP=UAV2_FOVP, UAV3_FOVP=UAV3_FOVP, UAV1_W_FOVP=UAV1_W_FOVP,
         UAV2_W_FOVP=UAV2_W_FOVP, UAV3_W_FOVP=UAV3_W_FOVP, targetarr=targetarr, UAV3_FOV_Plot_RL=UAV3_FOV_Plot_RL,
         UAV2_FOV_Plot_RL=UAV2_FOV_Plot_RL, UAV1_FOV_Plot_RL=UAV1_FOV_Plot_RL)


# Optimal Vs Wrl Plot
fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.plot(times1, minerrorarr[0:max_steps], color="blue")
plt.plot(times1, final_err_arr[0:max_steps], color="brown")
plt.legend(loc=4)
plt.legend(['Error of Optimal Policy (With RL)', 'Error without RL'])
plt.title('Error Plot')  # Title of the plot
plt.xlabel('Time (s)')  # X-Label
plt.ylabel('Error (m)')  # Y-Label

fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.bar(x_ax_uav_f, y_ax_uav_f)
plt.title('Error Over Entire Lap')  # Title of the plot


# UAV 1 RL

fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1, controls_s[0, 0:max_steps], linewidth="2", color="brown")
plt.plot(times1, linear_acc_min, '--', color="red")
plt.plot(times1, linear_acc_max, '--', color="red")
plt.title('Linear Acceleration of Multirotor')
plt.xlabel('Time (s)')
plt.ylabel('m/s')
fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1, controls_s[1, 0:max_steps], linewidth="2", color="blue")
plt.plot(times1, pitch_rate_min, '--', color="red")
plt.plot(times1, pitch_rate_max, '--', color="red")
plt.title('Pitch Rate of Multirotor')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')
fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1, controls_s[2, 0:max_steps], linewidth="2", color="black")
plt.plot(times1, yaw_rate_min, '--', color="red")
plt.plot(times1, yaw_rate_max, '--', color="red")
plt.title('Yaw Rate of Multirotor')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')
fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1, controls_s[3, 0:max_steps], linewidth="2", color="grey")
plt.plot(times1, groll_rate_min, '--', color="red")
plt.plot(times1, groll_rate_max, '--', color="red")
plt.title('Gimbal Roll Rate')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')
fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1, controls_s[4, 0:max_steps], linewidth="2", color="green")
plt.plot(times1, gpitch_rate_min, '--', color="red")
plt.plot(times1, gpitch_rate_max, '--', color="red")
plt.title('Gimbal Pitch Rate')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')
fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1, controls_s[5, 0:max_steps], linewidth="2", color="purple")
plt.plot(times1, gyaw_rate_min, '--', color="red")
plt.plot(times1, gyaw_rate_max, '--', color="red")
plt.title('Gimbal Yaw Rate')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')

# UAV 2 RL
fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1, controls_s[0+6, 0:max_steps], linewidth="2", color="brown")
plt.plot(times1, linear_acc_min_1, '--', color="red")
plt.plot(times1, linear_acc_max_1, '--', color="red")
plt.title('Linear Acceleration of FW-AAV-1')
plt.xlabel('Time (s)')
plt.ylabel('m/s')
fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1, controls_s[1+6, 0:max_steps], linewidth="2", color="blue")
plt.plot(times1, pitch_rate_min, '--', color="red")
plt.plot(times1, pitch_rate_max, '--', color="red")
plt.title('Pitch Rate of FW-AAV-1')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')
fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1, controls_s[2+6, 0:max_steps], linewidth="2", color="black")
plt.plot(times1, yaw_rate_min_1, '--', color="red")
plt.plot(times1, yaw_rate_max_1, '--', color="red")
plt.title('Yaw Rate of FW-AAV-1')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')
fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1, controls_s[3+6, 0:max_steps], linewidth="2", color="grey")
plt.plot(times1, groll_rate_min, '--', color="red")
plt.plot(times1, groll_rate_max, '--', color="red")
plt.title('Gimbal Roll Rate')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')
fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1, controls_s[4+6, 0:max_steps], linewidth="2", color="green")
plt.plot(times1, gpitch_rate_min, '--', color="red")
plt.plot(times1, gpitch_rate_max, '--', color="red")
plt.title('Gimbal Pitch Rate')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')
fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1, controls_s[5+6, 0:max_steps], linewidth="2", color="purple")
plt.plot(times1, gyaw_rate_min, '--', color="red")
plt.plot(times1, gyaw_rate_max, '--', color="red")
plt.title('Gimbal Yaw Rate')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')

# UAV3 RL

fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1, controls_s[0+6+6, 0:max_steps], linewidth="2", color="brown")
plt.plot(times1, linear_acc_min_2, '--', color="red")
plt.plot(times1, linear_acc_max_2, '--', color="red")
plt.title('Linear Acceleration of FW-AAV-2')
plt.xlabel('Time (s)')
plt.ylabel('m/s')
fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1, controls_s[1+6+6, 0:max_steps], linewidth="2", color="blue")
plt.plot(times1, pitch_rate_min, '--', color="red")
plt.plot(times1, pitch_rate_max, '--', color="red")
plt.title('Pitch Rate of FW-AAV-2')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')
fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1, controls_s[2+6+6, 0:max_steps], linewidth="2", color="black")
plt.plot(times1, yaw_rate_min_2, '--', color="red")
plt.plot(times1, yaw_rate_max_2, '--', color="red")
plt.title('Yaw Rate of FW-AAV-2')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')
fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1, controls_s[3+12, 0:max_steps], linewidth="2", color="grey")
plt.plot(times1, groll_rate_min, '--', color="red")
plt.plot(times1, groll_rate_max, '--', color="red")
plt.title('Gimbal Roll Rate')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')
fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1, controls_s[4+12, 0:max_steps], linewidth="2", color="green")
plt.plot(times1, gpitch_rate_min, '--', color="red")
plt.plot(times1, gpitch_rate_max, '--', color="red")
plt.title('Gimbal Pitch Rate')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')
fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1, controls_s[5+12, 0:max_steps], linewidth="2", color="purple")
plt.plot(times1, gyaw_rate_min, '--', color="red")
plt.plot(times1, gyaw_rate_max, '--', color="red")
plt.title('Gimbal Yaw Rate')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')

#
# # UAV 1 Without RL
# fig = plt.figure()
# plt.rcParams.update({'font.size': 50})
# plt.plot(times1, controls_o[0, 0:max_steps], linewidth="2", color="brown")
# plt.plot(times1, linear_vel_min, '--', color="red")
# plt.plot(times1, linear_vel_max, '--', color="red")
# plt.title('Linear Velocity of Drone')
# plt.xlabel('Time (s)')
# plt.ylabel('m/s')
# fig = plt.figure()
# plt.rcParams.update({'font.size': 50})
# plt.plot(times1, controls_o[1, 0:max_steps], linewidth="2", color="blue")
# plt.plot(times1, pitch_rate_min, '--', color="red")
# plt.plot(times1, pitch_rate_max, '--', color="red")
# plt.title('Pitch Rate of Drone')
# plt.xlabel('Time (s)')
# plt.ylabel('rad/s')
# fig = plt.figure()
# plt.rcParams.update({'font.size': 50})
# plt.plot(times1, controls_o[2, 0:max_steps], linewidth="2", color="black")
# plt.plot(times1, yaw_rate_min, '--', color="red")
# plt.plot(times1, yaw_rate_max, '--', color="red")
# plt.title('Yaw Rate of Drone')
# plt.xlabel('Time (s)')
# plt.ylabel('rad/s')
# fig = plt.figure()
# plt.rcParams.update({'font.size': 50})
# plt.plot(times1, controls_o[3, 0:max_steps], linewidth="2", color="grey")
# plt.plot(times1, groll_rate_min, '--', color="red")
# plt.plot(times1, groll_rate_max, '--', color="red")
# plt.title('Gimbal Roll Rate')
# plt.xlabel('Time (s)')
# plt.ylabel('rad/s')
# fig = plt.figure()
# plt.rcParams.update({'font.size': 50})
# plt.plot(times1, controls_o[4, 0:max_steps], linewidth="2", color="green")
# plt.plot(times1, gpitch_rate_min, '--', color="red")
# plt.plot(times1, gpitch_rate_max, '--', color="red")
# plt.title('Gimbal Pitch Rate')
# plt.xlabel('Time (s)')
# plt.ylabel('rad/s')
# fig = plt.figure()
# plt.rcParams.update({'font.size': 50})
# plt.plot(times1, controls_o[5, 0:max_steps], linewidth="2", color="purple")
# plt.plot(times1, gyaw_rate_min, '--', color="red")
# plt.plot(times1, gyaw_rate_max, '--', color="red")
# plt.title('Gimbal Yaw Rate')
# plt.xlabel('Time (s)')
# plt.ylabel('rad/s')
#
#
# # UAV 2 Without RL
# fig = plt.figure()
# plt.rcParams.update({'font.size': 50})
# plt.plot(times1, controls_o[0+6, 0:max_steps], linewidth="2", color="brown")
# plt.plot(times1, linear_vel_min_1, '--', color="red")
# plt.plot(times1, linear_vel_max, '--', color="red")
# plt.title('Linear Velocity of AAV-1')
# plt.xlabel('Time (s)')
# plt.ylabel('m/s')
# fig = plt.figure()
# plt.rcParams.update({'font.size': 50})
# plt.plot(times1, controls_o[1+6, 0:max_steps], linewidth="2", color="blue")
# plt.plot(times1, pitch_rate_min, '--', color="red")
# plt.plot(times1, pitch_rate_max, '--', color="red")
# plt.title('Pitch Rate of AAV-1')
# plt.xlabel('Time (s)')
# plt.ylabel('rad/s')
# fig = plt.figure()
# plt.rcParams.update({'font.size': 50})
# plt.plot(times1, controls_o[2+6, 0:max_steps], linewidth="2", color="black")
# plt.plot(times1, yaw_rate_min, '--', color="red")
# plt.plot(times1, yaw_rate_max, '--', color="red")
# plt.title('Yaw Rate of AAV-1')
# plt.xlabel('Time (s)')
# plt.ylabel('rad/s')
# fig = plt.figure()
# plt.rcParams.update({'font.size': 50})
# plt.plot(times1, controls_o[3+6, 0:max_steps], linewidth="2", color="grey")
# plt.plot(times1, groll_rate_min, '--', color="red")
# plt.plot(times1, groll_rate_max, '--', color="red")
# plt.title('Gimbal Roll Rate')
# plt.xlabel('Time (s)')
# plt.ylabel('rad/s')
# fig = plt.figure()
# plt.rcParams.update({'font.size': 50})
# plt.plot(times1, controls_o[4+6, 0:max_steps], linewidth="2", color="green")
# plt.plot(times1, gpitch_rate_min, '--', color="red")
# plt.plot(times1, gpitch_rate_max, '--', color="red")
# plt.title('Gimbal Pitch Rate')
# plt.xlabel('Time (s)')
# plt.ylabel('rad/s')
# fig = plt.figure()
# plt.rcParams.update({'font.size': 50})
# plt.plot(times1, controls_o[5+6, 0:max_steps], linewidth="2", color="purple")
# plt.plot(times1, gyaw_rate_min, '--', color="red")
# plt.plot(times1, gyaw_rate_max, '--', color="red")
# plt.title('Gimbal Yaw Rate')
# plt.xlabel('Time (s)')
# plt.ylabel('rad/s')
#
#
# # UAV 3 Without RL
# fig = plt.figure()
# plt.rcParams.update({'font.size': 50})
# plt.plot(times1, controls_o[0+6+6, 0:max_steps], linewidth="2", color="brown")
# plt.plot(times1, linear_vel_min_1, '--', color="red")
# plt.plot(times1, linear_vel_max, '--', color="red")
# plt.title('Linear Velocity of AAV-2')
# plt.xlabel('Time (s)')
# plt.ylabel('m/s')
# fig = plt.figure()
# plt.rcParams.update({'font.size': 50})
# plt.plot(times1, controls_o[1+6+6, 0:max_steps], linewidth="2", color="blue")
# plt.plot(times1, pitch_rate_min, '--', color="red")
# plt.plot(times1, pitch_rate_max, '--', color="red")
# plt.title('Pitch Rate of AAV-2')
# plt.xlabel('Time (s)')
# plt.ylabel('rad/s')
# fig = plt.figure()
# plt.rcParams.update({'font.size': 50})
# plt.plot(times1, controls_o[2+6+6, 0:max_steps], linewidth="2", color="black")
# plt.plot(times1, yaw_rate_min, '--', color="red")
# plt.plot(times1, yaw_rate_max, '--', color="red")
# plt.title('Yaw Rate of AAV-2')
# plt.xlabel('Time (s)')
# plt.ylabel('rad/s')
# fig = plt.figure()
# plt.rcParams.update({'font.size': 50})
# plt.plot(times1, controls_o[3+12, 0:max_steps], linewidth="2", color="grey")
# plt.plot(times1, groll_rate_min, '--', color="red")
# plt.plot(times1, groll_rate_max, '--', color="red")
# plt.title('Gimbal Roll Rate')
# plt.xlabel('Time (s)')
# plt.ylabel('rad/s')
# fig = plt.figure()
# plt.rcParams.update({'font.size': 50})
# plt.plot(times1, controls_o[4+12, 0:max_steps], linewidth="2", color="green")
# plt.plot(times1, gpitch_rate_min, '--', color="red")
# plt.plot(times1, gpitch_rate_max, '--', color="red")
# plt.title('Gimbal Pitch Rate')
# plt.xlabel('Time (s)')
# plt.ylabel('rad/s')
# fig = plt.figure()
# plt.rcParams.update({'font.size': 50})
# plt.plot(times1, controls_o[5+12, 0:max_steps], linewidth="2", color="purple")
# plt.plot(times1, gyaw_rate_min, '--', color="red")
# plt.plot(times1, gyaw_rate_max, '--', color="red")
# plt.title('Gimbal Yaw Rate')
# plt.xlabel('Time (s)')
# plt.ylabel('rad/s')

# plottting UAV and target with and without RL
UAV_RL = np.array(UAV_RL)
UAV_W_RL = np.array(UAV_W_RL)
targetarr = np.array(targetarr)

fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.plot(times1, UAV_RL[2, 0:max_steps], color="blue")
plt.plot(times1, UAV_W_RL[2, 0:max_steps], color="red")
plt.plot(times1, z_aav_min, '--', color="brown")
plt.plot(times1, z_aav_max, '--', color="brown")
plt.xlabel('Time (s)')
plt.ylabel('Height (m)')
plt.title('Height of AAV')
plt.legend(['AAV With RL', 'AAV Without RL'])

fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.plot(times1, w1, linewidth="2", color="blue")
plt.title('Weight-1 Trajectory')
plt.xlabel('Time (s)')
plt.ylabel('Weight')

fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.plot(times1, w2, linewidth="2", color="green")
plt.title('Weight-2 Trajectory')
plt.xlabel('Time (s)')
plt.ylabel('Weight')

fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.plot(times1, w3, linewidth="2", color="blue")
plt.title('Weight-3 Trajectory')
plt.xlabel('Time (s)')
plt.ylabel('Weight')

fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.plot(times1, w4, linewidth="2", color="green")
plt.title('Weight-4 Trajectory')
plt.xlabel('Time (s)')
plt.ylabel('Weight')

fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.plot(times1, w5, linewidth="2", color="blue")
plt.title('Weight-5 Trajectory')
plt.xlabel('Time (s)')
plt.ylabel('Weight')

fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.plot(times1, w6, linewidth="2", color="green")
plt.title('Weight-6 Trajectory')
plt.xlabel('Time (s)')
plt.ylabel('Weight')

## Error Plots
fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.plot(times1, error1_uni[0:max_steps], color="red")
plt.plot(times1, UAV1_RL_Err[0:max_steps], color="blue")
plt.legend(['Without RL', 'With RL'])
plt.title('Error Plot Drone')  # Title of the plot
plt.xlabel('Time (s)')  # X-Label
plt.ylabel('Error (m)')  # Y-Label
fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.bar(x_ax_uav1, y_ax_uav1)

fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.plot(times1, error2_uni[0:max_steps], color="red")
plt.plot(times1, UAV2_RL_Err[0:max_steps], color="blue")
plt.legend(['Without RL', 'With RL'])
plt.title('Error Plot FW-UAV 1')  # Title of the plot
plt.xlabel('Time (s)')  # X-Label
plt.ylabel('Error (m)')  # Y-Label
fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.bar(x_ax_uav2, y_ax_uav2)

fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.plot(times1, error3_uni[0:max_steps], color="red")
plt.plot(times1, UAV2_RL_Err[0:max_steps], color="blue")
plt.legend(['Without RL', 'With RL'])
plt.title('Error Plot FW-UAV 2')  # Title of the plot
plt.xlabel('Time (s)')  # X-Label
plt.ylabel('Error (m)')  # Y-Label
fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.bar(x_ax_uav3, y_ax_uav3)

fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.plot(times1, final_err_arr[0:max_steps], color="red")
plt.plot(times1, minerrorarr[0:max_steps], color="blue")
plt.legend(['Without RL', 'With RL'])
plt.title('Total Error Plot')  # Title of the plot
plt.xlabel('Time (s)')  # X-Label
plt.ylabel('Error (m)')  # Y-Label
fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.bar(x_ax_uav_f, y_ax_uav_f)


# # fig = plt.figure()
# # ax = plt.axes(projection='3d')
# # ax.plot3D(UAV_RL[0, 0:max_steps], UAV_RL[1, 0:max_steps], UAV_RL[2, 0:max_steps], linewidth="2", color="brown")
# # ax.plot3D(UAV_W_RL[0, 0:max_steps], UAV_W_RL[1, 0:max_steps], UAV_W_RL[2, 0:max_steps], linewidth="2", color="blue")
# # ax.plot3D(targetarr[0, 0:max_steps], targetarr[1, 0:max_steps], ss1[0:max_steps], '--', linewidth="2", color="red")
# # ax.plot3D(UAV_RL[0, 0:max_steps], UAV_RL[1, 0:max_steps], ss1[0:max_steps], linewidth="2", color="pink")
# # ax.plot3D(UAV_W_RL[0, 0:max_steps], UAV_W_RL[1, 0:max_steps], ss1[0:max_steps], linewidth="2", color="green")
# # ax.plot3D(FOV_C_RL[0:max_step_size, 0], FOV_C_RL[0:max_step_size, 1], ss1[0:max_steps], linewidth="2", color="grey")
# # ax.plot3D(FOV_C_W_RL[0:max_step_size, 0], FOV_C_W_RL[0:max_step_size, 1], ss1[0:max_steps], linewidth="2",
# #           color="purple")
# # ax.plot3D(x_e_rl[0:100, 0], y_e_rl[0:100, 0], ss1[0:100], linewidth="2", color="black")
# # ax.plot3D(x_e_w_rl[0:100, 0], y_e_w_rl[0:100, 0], ss1[0:100], linewidth="2", color="orange")
# # # ax.plot3D(x_e_2[0:100, 0], y_e_2[0:100, 0] , ss1[0:100], linewidth = "2", color = "black")
# # # ax.plot_surface(Xc_1, Yc_1, Zc_1, cstride = 1,rstride= 1)  # 0bstacles
# # # ax.plot_surface(Xc_2, Yc_2, Zc_2)
# # # ax.plot_surface(Xc_3, Yc_3, Zc_3)
# # ax.set_title('UAV trajectories with and without RL')
# # ax.set_xlabel('x-axis')
# # ax.set_ylabel('y-axis')
# # ax.set_zlabel('z-axis')
# # ax.legend(['UAV With RL', 'UAV Without RL', 'Target', 'Projection of UAV With RL', 'Projection of UAV Without RL',
# #            'Center of FOV with RL', 'Center of FOV without RL', 'RL FOV', 'Without RL FOV'])
#
# # 3D figure plotting by using mayavi final plotting of simulation
fig1 = mlab.figure()
mlab.clf()  # Clear the figure
mlab.plot3d(UAV_RL[0, 0:max_steps], UAV_RL[1, 0:max_steps], UAV_RL[2, 0:max_steps], tube_radius=5,
            color=(255 / 255, 166 / 255, 0))
mlab.plot3d(UAV_RL[8, 0:max_steps], UAV_RL[1+8, 0:max_steps], UAV_RL[2+8, 0:max_steps], tube_radius=5,
            color=(255 / 255, 166 / 255, 0))
mlab.plot3d(UAV_RL[0+16, 0:max_steps], UAV_RL[1+16, 0:max_steps], UAV_RL[2+16, 0:max_steps], tube_radius=5,
            color=(255 / 255, 166 / 255, 0))
mlab.plot3d(targetarr[0, 0:max_steps], targetarr[1, 0:max_steps], ss1[0:max_steps], tube_radius=5, color=(0, .7, 0))
mlab.plot3d(targetarr[0+3, 0:max_steps], targetarr[1+3, 0:max_steps], ss1[0:max_steps], tube_radius=5, color=(0, .7, 0))
mlab.plot3d(targetarr[0+6, 0:max_steps], targetarr[1+6, 0:max_steps], ss1[0:max_steps], tube_radius=5, color=(0, .7, 0))
mlab.plot3d(UAV_RL[0, 0:max_steps], UAV_RL[1, 0:max_steps], ss1[0:max_steps], tube_radius=3,
            color=(255 / 255, 181 / 255, 187 / 255))
mlab.plot3d(FOV_C_RL[0:max_step_size, 0], FOV_C_RL[0:max_step_size, 1], ss1[0:max_steps], tube_radius=3,
            color=(125 / 255, 164 / 255, 208 / 255))
mlab.plot3d(x_e_rl[0:100, 0], y_e_rl[0:100, 0], ss1[0:100], tube_radius=3, color=(0, 0, 0))
mlab.mesh(Xc_1, Yc_1, Zc_1)
mlab.mesh(Xc_2, Yc_2, Zc_2)
mlab.mesh(Xc_3, Yc_3, Zc_3)
mlab.mesh(Xc_4, Yc_4, Zc_4)
mlab.mesh(Xc_5, Yc_5, Zc_5)
mlab.mesh(Xc_6, Yc_6, Zc_6)
mlab.mesh(Xc_7, Yc_7, Zc_7)
mlab.mesh(Xc_8, Yc_8, Zc_8)
mlab.title('Tracking UAV')
mlab.orientation_axes(xlabel='X-axis', ylabel='Y-axis', zlabel='Z-axis')
mlab.show()

# new velocity
fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1, UAV_RL[24, 0:max_step_size], linewidth="2", color="brown")
plt.plot(times1, linear_vel_min, '--', color="red")
plt.plot(times1, linear_vel_max, '--', color="red")
plt.title('Linear Velocity of Multirotor')
plt.xlabel('Time (s)')
plt.ylabel('m/s')

fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1[0:max_step_size-7], UAV_RL[25, 7:max_step_size], linewidth="2", color="brown")
plt.plot(times1[0:max_step_size-7], linear_vel_min_1[0:max_step_size-7], '--', color="red")
plt.plot(times1[0:max_step_size-7], linear_vel_max_1[0:max_step_size-7], '--', color="red")
plt.title('Linear Velocity of FW-AAV-1')
plt.xlabel('Time (s)')
plt.ylabel('m/s')

fig = plt.figure()
plt.rcParams.update({'font.size': 50})
plt.plot(times1[0:max_step_size-9], UAV_RL[26, 9:max_step_size], linewidth="2", color="brown")
plt.plot(times1[0:max_step_size-9], linear_vel_min_2[0:max_step_size-9], '--', color="red")
plt.plot(times1[0:max_step_size-9], linear_vel_max_2[0:max_step_size-9], '--', color="red")
plt.title('Linear Velocity of FW-AAV-2')
plt.xlabel('Time (s)')
plt.ylabel('m/s')

fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.plot(times1, dist4[0:max_step_size], linewidth="2", color="blue")
plt.plot(times1, dist5[0:max_step_size], linewidth="2", color="green")
plt.plot(times1, dist6[0:max_step_size], linewidth="2", color="red")
plt.legend(['Multirotor - FW-AAV-1', 'FW-AAV-1 - FW-AAV-2', 'FW-AAV-2 - Multirotor'])
plt.xlabel("Time (s)")
plt.ylabel("Distance b/w AAVs")

fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.bar(x_control_effort1, y_control_effort1)

fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.bar(x_control_effort2, y_control_effort2)

fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.bar(x_control_effort3, y_control_effort3)

fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.bar(x_control_effort4, y_control_effort4)

fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.bar(x_control_effort5, y_control_effort5)

fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.bar(x_control_effort6, y_control_effort6)

fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.bar(x_control_effort7, y_control_effort7)

fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.bar(x_control_effort8, y_control_effort8)

fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.bar(x_control_effort9, y_control_effort9)

fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.bar(x_control_effort10, y_control_effort10)

fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.bar(x_control_effort11, y_control_effort11)

fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.bar(x_control_effort12, y_control_effort12)

fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.bar(x_control_effort13, y_control_effort13)

fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.bar(x_control_effort14, y_control_effort14)

fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.bar(x_control_effort15, y_control_effort15)

fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.bar(x_control_effort16, y_control_effort16)

fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.bar(x_control_effort17, y_control_effort17)

fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.bar(x_control_effort18, y_control_effort18)


fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.subplot(3, 6, 1)
plt.bar(x_control_effort1, y_control_effort1)
plt.subplot(3, 6, 2)
plt.bar(x_control_effort2, y_control_effort2)
plt.subplot(3, 6, 3)
plt.bar(x_control_effort3, y_control_effort3)
plt.subplot(3, 6, 4)
plt.bar(x_control_effort4, y_control_effort4)
plt.subplot(3, 6, 5)
plt.bar(x_control_effort5, y_control_effort5)
plt.subplot(3, 6, 6)
plt.bar(x_control_effort6, y_control_effort6)
plt.subplot(3, 6, 7)
plt.bar(x_control_effort7, y_control_effort7)
plt.subplot(3, 6, 8)
plt.bar(x_control_effort8, y_control_effort8)
plt.subplot(3, 6, 9)
plt.bar(x_control_effort9, y_control_effort9)
plt.subplot(3, 6, 10)
plt.bar(x_control_effort10, y_control_effort10)
plt.subplot(3, 6, 11)
plt.bar(x_control_effort11, y_control_effort11)
plt.subplot(3, 6, 12)
plt.bar(x_control_effort12, y_control_effort12)
plt.subplot(3, 6, 13)
plt.bar(x_control_effort13, y_control_effort13)
plt.subplot(3, 6, 14)
plt.bar(x_control_effort14, y_control_effort14)
plt.subplot(3, 6, 15)
plt.bar(x_control_effort15, y_control_effort15)
plt.subplot(3, 6, 16)
plt.bar(x_control_effort16, y_control_effort16)
plt.subplot(3, 6, 17)
plt.bar(x_control_effort17, y_control_effort17)
plt.subplot(3, 6, 18)
plt.bar(x_control_effort18, y_control_effort18)

plt.show()