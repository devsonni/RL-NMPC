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
x0 = [99, 150, 80, 0, 0, 0, 0, 0]  # initial UAV states
xs = [100, 150, 0]  # initial target state
mpc_iter = 0  # initial MPC count
max_step_size = 2000
mpc = 0
sc = 0
ss1 = np.zeros((max_step_size+1000))
controls_s = ca.DM.zeros((6,max_step_size))
controls_o = ca.DM.zeros((6,max_step_size))
x_o_1 = 0
y_o_1 = 80
x_o_2 = 500
y_o_2 = 245
x_o_3 = 1000
y_o_3 = 70
x_o_4 = 1500
y_o_4 = 295
x_o_5 = 1765
y_o_5 = 550
x_o_6 = 1500
y_o_6 = 750
x_o_7 = 1000
y_o_7 = 1005
x_o_8 = 500
y_o_8 = 800
x_o_9 = -100
y_o_9 = 950
x_o_10 = -200
y_o_10 = 550
obs_r = 50
UAV_r = 5

# function to shift MPC one step ahead
def shift_timestep(T, t0, x0, u, f_u, xs):
    f_value = f_u(x0, u[:, 0])
    x0 = ca.DM.full(x0 + (T * f_value))

    t0 = t0 + T
    u0 = ca.horzcat(
        u[:, 1:],
        ca.reshape(u[:, -1], -1, 1)
    )

    global sc
    v = 12
    con_t = [v, 0]  # Linear and angular velocity of target
    if (sc >= 500):
        con_t = [v, (pi) / 100]  # right turn
    if (sc >= 1000):
        con_t = [v, 0]  # 50 ahead
    if (sc >= 1500):
        con_t = [v, (pi) / 100]  # left ahead

    f_t_value = ca.vertcat(con_t[0] * cos(xs[2]),
                           con_t[0] * sin(xs[2]),
                           con_t[1])
    xs = ca.DM.full(xs + (T * f_t_value))
    return t0, x0, u0, xs


# For plotting a cylinder
def data_for_cylinder_along_z(center_x,center_y,radius,height_z):
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid

# convert DM and SX to array
def DM2Arr(dm):
    return np.array(dm.full())

def SX2Arr(sx):
    return np.array(sx.full())

# For plotting ellipse
def ellipse(a_p, b_p, x_e_1, y_e_1):
    x = ca.DM.zeros((101))
    y = ca.DM.zeros((101))
    th = np.linspace(0,  2*np.pi, 100)
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
def MPC(w1, w2):
    # mpc parameters
    T = 0.2  # discrete step
    N = 15  # number of look ahead steps

    # Constrains of UAV with gimbal
    # input constrains of UAV
    v_u_min = 14
    v_u_max = 30
    omega_2_u_min = -pi / 30
    omega_2_u_max = pi / 30
    omega_3_u_min = -pi / 21
    omega_3_u_max = pi / 21

    # input constrains of gimbal
    omega_1_g_min = -pi / 30
    omega_1_g_max = pi / 30
    omega_2_g_min = -pi / 30
    omega_2_g_max = pi / 30
    omega_3_g_min = -pi / 30
    omega_3_g_max = pi / 30

    # states constrains of UAV
    theta_u_min = -0.2618
    theta_u_max = 0.2618
    z_u_min = 75
    z_u_max = 150

    # states constrains of gimbal
    phi_g_min = -pi / 6
    phi_g_max = pi / 6
    theta_g_min = -pi / 6
    theta_g_max = pi / 6
    shi_g_min = -pi / 2
    shi_g_max = pi / 2

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
    )
    n_states_u = states_u.numel()

    # Controls of UAV that will find by NMPC
    # UAV controls parameters
    v_u = ca.SX.sym('v_u')
    omega_2_u = ca.SX.sym('omega_2_u')
    omega_3_u = ca.SX.sym('omega_3_u')
    # Gimbal control parameters
    omega_1_g = ca.SX.sym('omega_1_g')
    omega_2_g = ca.SX.sym('omega_2_g')
    omega_3_g = ca.SX.sym('omega_3_g')

    # Appending controls in one vector
    controls_u = ca.vertcat(
        v_u,
        omega_2_u,
        omega_3_u,
        omega_1_g,
        omega_2_g,
        omega_3_g,
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
        omega_3_g
    )

    # Non-linear mapping function which is f(x,y)
    f_u = ca.Function('f', [states_u, controls_u], [rhs_u])

    U = ca.SX.sym('U', n_controls, N)  # Decision Variables
    P = ca.SX.sym('P', n_states_u + 3)  # This consists of initial states of UAV with gimbal 1-8 and
    # reference states 9-11 (reference states is target's states)

    X = ca.SX.sym('X', n_states_u, (N + 1))  # Has prediction of states over prediction horizon

    # Filling the defined system parameters of UAV
    X[:, 0] = P[0:8]  # initial state

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

    for k in range(N):
        stt = X[0:8, 0:N]
        A = ca.SX.sym('A', N)
        B = ca.SX.sym('B', N)
        C = ca.SX.sym('C', N)
        X_E = ca.SX.sym('X_E', N)
        Y_E = ca.SX.sym('Y_E', N)

        VFOV = 1  # Making FOV
        HFOV = 1

        #w1 = 1  # MPC weights
        #w2 = 2

        a = (stt[2, k] * (tan(stt[6, k] + VFOV / 2)) - stt[2, k] * tan(stt[6, k] - VFOV / 2)) / 2  # FOV Stuff
        b = (stt[2, k] * (tan(stt[5, k] + HFOV / 2)) - stt[2, k] * tan(stt[5, k] - HFOV / 2)) / 2

        A[k] = ((cos(stt[7, k])) ** 2) / a ** 2 + ((sin(stt[7, k])) ** 2) / b ** 2
        B[k] = 2 * cos(stt[7, k]) * sin(stt[7, k]) * ((1 / a ** 2) - (1 / b ** 2))
        C[k] = ((sin(stt[7, k])) ** 2) / a ** 2 + ((cos(stt[7, k])) ** 2) / b ** 2

        X_E[k] = stt[0, k] + a + stt[2, k] * (tan(stt[6, k] - VFOV / 2))  # Centre of FOV
        Y_E[k] = stt[1, k] + b + stt[2, k] * (tan(stt[5, k] - HFOV / 2))

        obj = obj + w1 * ca.sqrt((stt[0, k] - P[8]) ** 2 + (stt[1, k] - P[9]) ** 2) + \
              w2 * ((A[k] * (P[8] - X_E[k]) ** 2 + B[k] * (P[9] - Y_E[k]) * (P[8] - X_E[k]) + C[k] * (P[9] - Y_E[k]) ** 2) - 1)

    # Obstacle parameters and virtual radius of uav
    global x_o_1, y_o_1, obs_r, x_o_2, y_o_2, x_o_3, y_o_3, UAV_r

    # compute the constrains, states or inequality constrains
    for k in range(N + 1):
        g = ca.vertcat(g,
                       X[2, k],  # limit on hight of UAV
                       X[3, k],  # limit on pitch angle theta
                       X[5, k],  # limit on gimbal angle phi
                       X[6, k],  # limit on gimbal angle theta
                       X[7, k],  # limit on gimbal angle shi
                       -ca.sqrt((X[0, k] - x_o_1) ** 2 + (X[1, k] - y_o_1) ** 2) + (UAV_r + obs_r),
                       # limit of obstacle-1
                       -ca.sqrt((X[0, k] - x_o_2) ** 2 + (X[1, k] - y_o_2) ** 2) + (UAV_r + obs_r),
                       # limit of obstacle-2
                       -ca.sqrt((X[0, k] - x_o_3) ** 2 + (X[1, k] - y_o_3) ** 2) + (UAV_r + obs_r),
                       # limit of obstacle-3
                       -ca.sqrt((X[0, k] - x_o_4) ** 2 + (X[1, k] - y_o_4) ** 2) + (UAV_r + obs_r),
                       # limit of obstacle-4
                       -ca.sqrt((X[0, k] - x_o_5) ** 2 + (X[1, k] - y_o_5) ** 2) + (UAV_r + obs_r),
                       # limit of obstacle-5
                       -ca.sqrt((X[0, k] - x_o_6) ** 2 + (X[1, k] - y_o_6) ** 2) + (UAV_r + obs_r),
                       # limit of obstacle-6
                       -ca.sqrt((X[0, k] - x_o_7) ** 2 + (X[1, k] - y_o_7) ** 2) + (UAV_r + obs_r),
                       # limit of obstacle-7
                       -ca.sqrt((X[0, k] - x_o_8) ** 2 + (X[1, k] - y_o_8) ** 2) + (UAV_r + obs_r),
                       # limit of obstacle-8
                       -ca.sqrt((X[0, k] - x_o_9) ** 2 + (X[1, k] - y_o_9) ** 2) + (UAV_r + obs_r),
                       # limit of obstacle-9
                       -ca.sqrt((X[0, k] - x_o_10) ** 2 + (X[1, k] - y_o_10) ** 2) + (UAV_r + obs_r)
                       # limit of obstacle-10
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
            'max_iter': 100,
            'print_level': 0,
            'acceptable_tol': 1e-8,
            'acceptable_obj_change_tol': 1e-6
        },
        'print_time': 0
    }

    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    lbx = ca.DM.zeros((n_controls * N, 1))
    ubx = ca.DM.zeros((n_controls * N, 1))
    lbg = ca.DM.zeros(((n_states_u + 7) * (N + 1)))
    ubg = ca.DM.zeros(((n_states_u + 7) * (N + 1)))

    # Constrains on states (Inequality constrains)
    lbg[0:240:15] = z_u_min  # z lower bound
    lbg[1:240:15] = theta_u_min  # theta lower bound
    lbg[2:240:15] = phi_g_min  # phi lower bound
    lbg[3:240:15] = theta_g_min  # theta lower bound
    lbg[4:240:15] = shi_g_min  # shi lower bound
    lbg[6:240:15] = -ca.inf  # Obstacle - 1
    lbg[5:240:15] = -ca.inf  # Obstacle - 2
    lbg[7:240:15] = -ca.inf  # Obstacle - 3
    lbg[8:240:15] = -ca.inf  # Obstacle - 4
    lbg[9:240:15] = -ca.inf  # Obstacle - 5
    lbg[10:240:15] = -ca.inf  # Obstacle - 6
    lbg[11:240:15] = -ca.inf  # Obstacle - 7
    lbg[12:240:15] = -ca.inf  # Obstacle - 8
    lbg[13:240:15] = -ca.inf  # Obstacle - 9
    lbg[14:240:15] = -ca.inf  # Obstacle - 10

    ubg[0:240:15] = z_u_max  # z lower bound
    ubg[1:240:15] = theta_u_max  # theta lower bound
    ubg[2:240:15] = phi_g_max  # phi lower bound
    ubg[3:240:15] = theta_g_max  # theta lower bound
    ubg[4:240:15] = shi_g_max  # shi lower bound
    ubg[5:240:15] = 0  # Obstacle - 1
    ubg[6:240:15] = 0  # Obstacle - 2
    ubg[7:240:15] = 0  # Obstacle - 3
    ubg[8:240:15] = 0  # Obstacle - 4
    ubg[9:240:15] = 0  # Obstacle - 5
    ubg[10:240:15] = 0  # Obstacle - 6
    ubg[11:240:15] = 0  # Obstacle - 7
    ubg[12:240:15] = 0  # Obstacle - 8
    ubg[13:240:15] = 0  # Obstacle - 9
    ubg[14:240:15] = 0  # Obstacle - 10

    # Constrains on controls, constrains on optimization variable
    lbx[0: n_controls * N: n_controls, 0] = v_u_min  # velocity lower bound
    lbx[1: n_controls * N: n_controls, 0] = omega_2_u_min  # theta 1 lower bound
    lbx[2: n_controls * N: n_controls, 0] = omega_3_u_min  # theta 2 lower bound
    lbx[3: n_controls * N: n_controls, 0] = omega_1_g_min  # omega 1 lower bound
    lbx[4: n_controls * N: n_controls, 0] = omega_2_g_min  # omega 2 lower bound
    lbx[5: n_controls * N: n_controls, 0] = omega_3_g_min  # omega 3 lower bound

    ubx[0: n_controls * N: n_controls, 0] = v_u_max  # velocity upper bound
    ubx[1: n_controls * N: n_controls, 0] = omega_2_u_max  # theta 1 upper bound
    ubx[2: n_controls * N: n_controls, 0] = omega_3_u_max  # theta 2 upper bound
    ubx[3: n_controls * N: n_controls, 0] = omega_1_g_max  # omega 1 upper bound
    ubx[4: n_controls * N: n_controls, 0] = omega_2_g_max  # omega 2 upper bound
    ubx[5: n_controls * N: n_controls, 0] = omega_3_g_max  # omega 3 upper bound

    args = {
        'lbg': lbg,  # lower bound for state
        'ubg': ubg,  # upper bound for state
        'lbx': lbx,  # lower bound for controls
        'ubx': ubx  # upper bound for controls
    }

    global x0, xs, mpc_iter, sc, controls_s, mpc
    t0 = 0

    # xx = DM(state_init)
    t = ca.DM(t0)

    u0 = ca.DM.zeros((n_controls, N))  # initial control
    xx = ca.DM.zeros((8, max_step_size))  # change size according to main loop run, works as tracker for target and UAV
    #ss = ca.DM.zeros((3, 801))
    x_e_1 = ca.DM.zeros((max_step_size))
    y_e_1 = ca.DM.zeros((max_step_size))

    #xx[:, 0] = x0
    #ss[:, 0] = xs

    mpc_iter = 0
    cat_controls = DM2Arr(u0[:, 0])
    times = np.array([[0]])

    ###############################################################################
    ##############################   Main Loop    #################################

    if __name__ == '__main__':
            t1 = time()
            ss[:, mpc_iter] = xs
            args['p'] = ca.vertcat(
                x0,  # current state
                xs  # target state
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

            cat_controls = np.vstack((
                cat_controls,
                DM2Arr(u[:, 0])
            ))

            #control_in = np.array(u[:,0])
            #print(control_in.shape)
            #print((controls_s[:,sc]).shape)
            if(mpc<max_step_size):
                controls_s[:,sc] = u[:,0]
            controls_o[:,sc] = u[:,0]
            # print(cat_controls)

            t = np.vstack((
                t,
                t0
            ))

            t0, x0, u0, xs = shift_timestep(T, t0, x0, u, f_u, xs)

            # tracking states of target and UAV for plotting
            xx[:, mpc_iter + 1] = x0

            # xx ...
            t2 = time()
            # print(t2-t1)
            times = np.vstack((
                times,
                t2 - t1
            ))

            mpc_iter = mpc_iter + 1
            sc = sc+1
            mpc = mpc+1

            a_p = (x0[2] * (tan(x0[6] + VFOV / 2)) - x0[2] * tan(x0[6] - VFOV / 2)) / 2  # For plotting FOV
            b_p = (x0[2] * (tan(x0[5] + HFOV / 2)) - x0[2] * tan(x0[5] - HFOV / 2)) / 2
            x_e_1[mpc_iter] = x0[0] + a_p + x0[2] * (tan(x0[6] - VFOV / 2))
            y_e_1[mpc_iter] = x0[1] + b_p + x0[2] * (tan(x0[5] - HFOV / 2))

            FOV_X = x_e_1[mpc_iter]
            FOV_Y = y_e_1[mpc_iter]

            error = ca.DM.zeros(max_step_size)

            Error = ca.sqrt((x_e_1[mpc_iter] - ss[0, mpc_iter-1]) ** 2 + (y_e_1[mpc_iter] - ss[1, mpc_iter-1]) ** 2)
            #print(Error)
    return Error, x0[0:3], xs, a_p, b_p, FOV_X, FOV_Y  # mpc functions returns error of specific iteration so agent can calculate reward


#################################################################################
############# Defining "tunning" reinforcement learning environment #############

class Tunning(Env):
    def __init__(self):
        # Action space which contains two discrete actions which are weights of MPC
        self.action_space = Tuple(spaces=(Discrete(101), Discrete(101)))
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
        error, obs, obs2, a_p, b_p, x_e_1, y_e_1 = MPC(action[0], action[1])
        reward = 1/error
        #print(error)
        print(reward)

        # Check if episode is done or not
        if self.episode_length <= 0:
            done = True
        else:
            done = False

        # extra info about env
        info = {}

        # Return step information
        return obs, obs2, reward, done, info, error, a_p, b_p, x_e_1, y_e_1

    def render(self):
        # Implement viz
        pass

    def reset(self):
        # Reset UAV & target to initial position
        global x0, xs, mpc_iter, max_step_size, ss, sc
        x0 = [99, 150, 80, 0, 0, 0, 0, 0]
        xs = [100, 150, 0]
        mpc_iter = 0
        sc = 0
        ss = ca.DM.zeros((3, max_step_size))
        self.episode_length = max_step_size


env = Tunning()
#################################################################
########################### Q-learning ##########################
step_size = max_step_size+1  # Change according to main loop run

qtable = np.zeros((step_size, 101, 101))

# Q - learning parameters
total_episodes = 500  # Total episodes
max_steps = max_step_size  # Max steps per episode


# List and array of rewards, errors etc.
rewards = []
rewardarr = np.zeros((total_episodes, max_steps))
errorarr = np.zeros((total_episodes, max_steps))
erroravg = np.zeros((total_episodes))
rewardavg = np.zeros((total_episodes))
maxreward = np.zeros((max_steps))
minerrorarr = np.zeros((max_steps))
action_str = np.zeros((max_steps + 1, 2))
w_1 = np.zeros((max_steps, total_episodes))
w_2 = np.zeros((max_steps, total_episodes))
UAV_RL = ca.DM.zeros((3,max_step_size+1))
UAV_W_RL = ca.DM.zeros((3,max_step_size+1))
targetarr = ca.DM.zeros((3,max_step_size+1))
error_w_rl = np.zeros((max_steps))
FOV_C_RL = ca.DM.zeros((max_step_size, 2))
FOV_C_W_RL = ca.DM.zeros((max_step_size, 2))

w1 = np.zeros((max_step_size))
w2 = np.zeros((max_step_size))
w3 = np.zeros((max_step_size))
w4 = np.zeros((max_step_size))
times = np.zeros((max_step_size))

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
arr = np.load("Race track 2_ 500epi 095alp 085gamma 00decay.npz")
errorarr = arr['errorarr']
total_episodes = arr['total_episodes']
w_1 = arr['w_1']
w_2 = arr['w_2']
rewardarr = arr['rewardarr']
erroravg = arr['erroravg']
rewardavg = arr['rewardavg']
qtable = arr['qtable']

#Collecing data of obstacles for plotting cylinders
Xc_1,Yc_1,Zc_1 = data_for_cylinder_along_z(x_o_1,y_o_1,obs_r,80)
Xc_2,Yc_2,Zc_2 = data_for_cylinder_along_z(x_o_2,y_o_2,obs_r,80)
Xc_3,Yc_3,Zc_3 = data_for_cylinder_along_z(x_o_3,y_o_3,obs_r,80)

for i in range(max_step_size):
    linear_vel_min[i] = 14
    linear_vel_max[i] = 30
    pitch_rate_min[i] = -pi/30
    pitch_rate_max[i] = pi/30
    yaw_rate_min[i] = -pi/21
    yaw_rate_max[i] = pi/21
    groll_rate_min[i] = -pi/30
    groll_rate_max[i] = pi/30
    gpitch_rate_min[i] = -pi/30
    gpitch_rate_max[i] = pi/30
    gyaw_rate_min[i] = -pi/30
    gyaw_rate_max[i] = pi/30
    z_aav_min[i] = 75
    z_aav_max[i] = 150
    times[i] = i * 0.2



Error = 0
env.reset()
UAV_RL[:, 0] = x0[0:3]
targetarr[:, 0] = xs
#Printing Optimal Policy
for i in range(max_step_size):
    action = max_index(qtable[i, :, :])
    w1[i] = action[0]
    w2[i] = action[1]
    w3[i] = 5
    w4[i] = 1
    new_state, obs2, reward, done, info, error, a_p, b_p, x_e_1, y_e_1 = env.step(action)
    FOV_C_RL[i, 0] = x_e_1
    FOV_C_RL[i, 1] = y_e_1
    UAV_RL[:, i+1] = new_state
    targetarr[:, i+1] = obs2
    maxreward[i] = reward
    minerrorarr[i] = error
    Error += error
    print(max_index(qtable[i, :, :]))

x_e_rl, y_e_rl = ellipse(a_p, b_p, x_e_1, y_e_1)
print(np.shape(x_e_rl))
x_e_rl = np.array(x_e_rl)
y_e_rl = np.array(y_e_rl)
FOV_C_RL = np.array(FOV_C_RL)
print('Error of Optimal Policy: {}'.format(Error))

# for without RL error and trajectory
Error = 0
env.reset()
UAV_W_RL[:, 0] = x0[0:3]
#Printing Optimal Policy
for i in range(max_step_size):
    action = (1, 1)
    new_state, obs2, reward, done, info, error, a_p, b_p, x_e_1, y_e_1 = env.step(action)
    FOV_C_W_RL[i, 0] = x_e_1
    FOV_C_W_RL[i, 1] = y_e_1
    UAV_W_RL[:, i+1] = new_state
    error_w_rl[i] = error
    Error += error
    #print(max_index(qtable[i, :, :]))
x_e_w_rl, y_e_w_rl = ellipse(a_p, b_p, x_e_1, y_e_1)
x_e_w_rl = np.array(x_e_w_rl)
y_e_w_rl = np.array(y_e_w_rl)
FOV_C_W_RL = np.array(FOV_C_W_RL)
print('Error without RL: {}'.format(Error))


################################################################
############# Plotting the reward & printing actions ###########
last_action = action_str[1:51, 0:2]
controls_s = np.array(controls_s)
controls_o = np.array(controls_o)
time = np.zeros((int(0.2*max_step_size)))
sum_without_rl = sum(error_w_rl)
sum_first_episode = sum(errorarr[0, 0:max_steps])
sum_intermediate_episode = sum(errorarr[(total_episodes//2)-1, 0:max_steps])
sum_last_episode = sum(errorarr[total_episodes-1, 0:max_steps])
sum_optimal_policy = sum(minerrorarr)
x_axis_bar = ['Error without RL', 'Error of OptimalPolicy (With RL)']
y_axis_bar = [sum_without_rl, sum_optimal_policy]
x_axis_bar_2 = ['Error of Fist Episode', 'Error of Imtermediate Episode', 'Error of Last Episode']
y_axis_bar_2 = [sum_first_episode, sum_intermediate_episode, sum_last_episode]

#sum_first_episode, sum_last_episode,-- 'Error of FirstEpisode', 'Error of LastEpisode',

#filling the time matrix
for i in range(int(0.2*max_step_size)):
    time[i] = i
# Printing actions
my_cmap = plt.get_cmap('cool')

fig = plt.figure()
plt.rcParams.update({'font.size':30})
plt.bar(x_axis_bar_2, y_axis_bar_2)
plt.title('Error Decrement Over Episodes')


# plotting actions evolving over episodes for w_1
fig = plt.figure(figsize=(14, 9))
plt.rcParams.update({'font.size':30})
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(x, y, w_1, cmap=my_cmap)
# Adding labels
ax.set_xlabel('Episodes')
ax.set_ylabel('Steps')
ax.set_zlabel('Weights (W1)')
ax.set_title('Weights-1 Evolving Over Episodes')

# plotting actions evolving over episodes for w_2
ax = fig.add_subplot(1, 2, 2, projection='3d')
plt.rcParams.update({'font.size':30})
ax.plot_surface(x, y, w_2, cmap=my_cmap)
# Adding labels
ax.set_xlabel('Episodes')
ax.set_ylabel('Steps')
ax.set_zlabel('Weights (W2)')
ax.set_title('Weights-2 Evolving Over Episodes')

# plotting rewards
fig = plt.figure()
plt.rcParams.update({'font.size':30})
plt.plot(rewardarr[0, 0:max_steps], color="blue")
plt.plot(rewardarr[total_episodes//2, 0:max_steps], color="green")
plt.plot(rewardarr[total_episodes-1, 0:max_steps], color="brown")
plt.plot(maxreward, color="red")
plt.legend(loc=4)
plt.legend(['Initial Episode', 'Intermediate Episode', 'Last Episode', 'Reward of Optimal Policy'])
plt.title('Rewards In Each Steps')  # Title of the plot
plt.xlabel('Steps')  # X-Label
plt.ylabel('Rewards')  # Y-Label

fig = plt.figure()
plt.rcParams.update({'font.size':30})
plt.plot(rewardavg, color="blue")
plt.title('Sum of Reward')  # Title of the plot
plt.xlabel('Episode')  # X-Label
plt.ylabel('Sum of Reward')  # Y-Label

#plotting error
fig = plt.figure()
plt.rcParams.update({'font.size':30})
plt.plot(erroravg, color="blue")
plt.title('Sum of Error')  # Title of the plot
plt.xlabel('Episode')  # X-Label
plt.ylabel('Sum of Error')  # Y-Label

fig = plt.figure()
plt.rcParams.update({'font.size':23})
plt.subplot(121)
plt.plot(times, minerrorarr[0:max_steps], color="blue")
plt.plot(times, error_w_rl[0:max_steps], color="brown")
plt.legend(loc=4)
plt.legend(['Error of Optimal Policy (With RL)', 'Error without RL'])
plt.title('Error Plot')  # Title of the plot
plt.xlabel('Time (s)')  # X-Label
plt.ylabel('Error (m)')  # Y-Label
plt.subplot(122)
plt.bar(x_axis_bar, y_axis_bar)

fig = plt.figure()
plt.rcParams.update({'font.size':50})
plt.plot(times, controls_s[0,0:max_steps], linewidth = "2", color = "brown")
plt.plot(times, linear_vel_min, '--', color = "red")
plt.plot(times, linear_vel_max, '--', color = "red")
plt.title('Linear Velocity of AAV')
plt.xlabel('Time (s)')
plt.ylabel('m/s')
fig = plt.figure()
plt.rcParams.update({'font.size':50})
plt.plot(times, controls_s[1,0:max_steps], linewidth = "2", color = "blue")
plt.plot(times, pitch_rate_min, '--', color = "red")
plt.plot(times, pitch_rate_max, '--', color = "red")
plt.title('Pitch Rate of AAV')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')
fig = plt.figure()
plt.rcParams.update({'font.size':50})
plt.plot(times, controls_s[2,0:max_steps], linewidth = "2", color = "black")
plt.plot(times, yaw_rate_min, '--', color = "red")
plt.plot(times, yaw_rate_max, '--', color = "red")
plt.title('Yaw Rate of AAV')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')
fig = plt.figure()
plt.rcParams.update({'font.size':50})
plt.plot(times, controls_s[3,0:max_steps], linewidth = "2", color = "blue")
plt.plot(times, groll_rate_min, '--', color = "red")
plt.plot(times, groll_rate_max, '--', color = "red")
plt.title('Gimbal Roll Rate')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')
fig = plt.figure()
plt.rcParams.update({'font.size':50})
plt.plot(times, controls_s[4,0:max_steps], linewidth = "2", color = "green")
plt.plot(times, gpitch_rate_min, '--', color = "red")
plt.plot(times, gpitch_rate_max, '--', color = "red")
plt.title('Gimbal Pitch Rate')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')
fig = plt.figure()
plt.rcParams.update({'font.size':50})
plt.plot(times, controls_s[5,0:max_steps], linewidth = "2", color = "purple")
plt.plot(times, gyaw_rate_min, '--', color = "red")
plt.plot(times, gyaw_rate_max, '--', color = "red")
plt.title('Gimbal Yaw Rate')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')

#without RL
fig = plt.figure()
plt.rcParams.update({'font.size':50})
plt.plot(times, controls_o[0,0:max_steps], linewidth = "2", color = "brown")
plt.plot(times, linear_vel_min, '--', color = "red")
plt.plot(times, linear_vel_max, '--', color = "red")
plt.title('Linear Velocity of AAV')
plt.xlabel('Time (s)')
plt.ylabel('m/s')
fig = plt.figure()
plt.rcParams.update({'font.size':50})
plt.plot(times, controls_o[1,0:max_steps], linewidth = "2", color = "blue")
plt.plot(times, pitch_rate_min, '--', color = "red")
plt.plot(times, pitch_rate_max, '--', color = "red")
plt.title('Pitch Rate of AAV')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')
fig = plt.figure()
plt.rcParams.update({'font.size':50})
plt.plot(times, controls_o[2,0:max_steps], linewidth = "2", color = "black")
plt.plot(times, yaw_rate_min, '--', color = "red")
plt.plot(times, yaw_rate_max, '--', color = "red")
plt.title('Yaw Rate of AAV')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')
fig = plt.figure()
plt.rcParams.update({'font.size':50})
plt.plot(times, controls_o[3,0:max_steps], linewidth = "2", color = "blue")
plt.plot(times, groll_rate_min, '--', color = "red")
plt.plot(times, groll_rate_max, '--', color = "red")
plt.title('Gimbal Roll Rate')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')
fig = plt.figure()
plt.rcParams.update({'font.size':50})
plt.plot(times, controls_o[4,0:max_steps], linewidth = "2", color = "green")
plt.plot(times, gpitch_rate_min, '--', color = "red")
plt.plot(times, gpitch_rate_max, '--', color = "red")
plt.title('Gimbal Pitch Rate')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')
fig = plt.figure()
plt.rcParams.update({'font.size':50})
plt.plot(times, controls_o[5,0:max_steps], linewidth = "2", color = "purple")
plt.plot(times, gyaw_rate_min, '--', color = "red")
plt.plot(times, gyaw_rate_max, '--', color = "red")
plt.title('Gimbal Yaw Rate')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')

#plottting UAV and target with and without RL
UAV_RL = np.array(UAV_RL)
UAV_W_RL = np.array(UAV_W_RL)
targetarr = np.array(targetarr)

fig =plt.figure()
plt.rcParams.update({'font.size':30})
plt.plot(times, UAV_RL[2, 0:max_steps], color = "blue")
plt.plot(times, UAV_W_RL[2, 0:max_steps], color = "red")
plt.plot(times, z_aav_min, '--', color = "brown")
plt.plot(times, z_aav_max, '--', color = "brown")
plt.xlabel('Time (s)')
plt.ylabel('Height (m)')
plt.title('Height of AAV')
plt.legend(['AAV With RL', 'AAV Without RL'])

fig = plt.figure()
plt.rcParams.update({'font.size':30})
plt.plot(times, w1, linewidth = "2", color = "blue")
plt.plot(times, w3, linewidth = "2", color = "red")
plt.legend(['Weight with RL', 'Weight without RL'])
plt.title('Weight-1 trajectory')
plt.xlabel('Time (s)')
plt.ylabel('Weight')

fig = plt.figure()
plt.rcParams.update({'font.size':30})
plt.plot(times, w2, linewidth = "2", color = "green")
plt.plot(times, w4, linewidth = "2", color = "black")
plt.legend(['Weight with RL', 'Weight without RL'])
plt.title('Weight-2 trajectory')
plt.xlabel('Time (s)')
plt.ylabel('Weight')

fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot3D(UAV_RL[0, 0:max_steps], UAV_RL[1, 0:max_steps], UAV_RL[2, 0:max_steps], linewidth = "2", color = "brown")
ax.plot3D(UAV_W_RL[0, 0:max_steps], UAV_W_RL[1, 0:max_steps], UAV_W_RL[2, 0:max_steps], linewidth = "2", color = "blue")
ax.plot3D(targetarr[0,0:max_steps], targetarr[1,0:max_steps], ss1[0:max_steps], '--',linewidth = "2", color = "red")
ax.plot3D(UAV_RL[0, 0:max_steps], UAV_RL[1, 0:max_steps], ss1[0:max_steps], linewidth = "2", color = "pink")
ax.plot3D(UAV_W_RL[0, 0:max_steps], UAV_W_RL[1, 0:max_steps], ss1[0:max_steps], linewidth = "2", color = "green")
ax.plot3D(FOV_C_RL[0:max_step_size, 0], FOV_C_RL[0:max_step_size, 1] , ss1[0:max_steps], linewidth = "2", color = "grey")
ax.plot3D(FOV_C_W_RL[0:max_step_size, 0], FOV_C_W_RL[0:max_step_size, 1] , ss1[0:max_steps], linewidth = "2", color = "purple")
ax.plot3D(x_e_rl[0:100, 0], y_e_rl[0:100, 0] , ss1[0:100], linewidth = "2", color = "black")
ax.plot3D(x_e_w_rl[0:100, 0], y_e_w_rl[0:100, 0] , ss1[0:100], linewidth = "2", color = "orange")
#ax.plot3D(x_e_2[0:100, 0], y_e_2[0:100, 0] , ss1[0:100], linewidth = "2", color = "black")
#ax.plot_surface(Xc_1, Yc_1, Zc_1, cstride = 1,rstride= 1)  # 0bstacles
#ax.plot_surface(Xc_2, Yc_2, Zc_2)
#ax.plot_surface(Xc_3, Yc_3, Zc_3)
ax.set_title('UAV trajectories with and without RL')
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')
ax.legend(['UAV With RL', 'UAV Without RL', 'Target', 'Projection of UAV With RL','Projection of UAV Without RL', 'Center of FOV with RL', 'Center of FOV without RL', 'RL FOV', 'Without RL FOV'])

#3D figure plotting by using mayavi final plotting of simulation
fig1 = mlab.figure()
mlab.clf()  # Clear the figure
mlab.plot3d(UAV_RL[0, 0:max_steps], UAV_RL[1, 0:max_steps], UAV_RL[2, 0:max_steps], tube_radius=5, color=(255/255, 166/255, 0))
mlab.plot3d(UAV_W_RL[0, 0:max_steps], UAV_W_RL[1, 0:max_steps], UAV_W_RL[2, 0:max_steps], tube_radius=5, color=(0,1,0))
mlab.plot3d(targetarr[0,0:max_steps], targetarr[1,0:max_steps], ss1[0:max_steps], tube_radius=5, color=(0,.7,0))
mlab.plot3d(UAV_RL[0, 0:max_steps], UAV_RL[1, 0:max_steps], ss1[0:max_steps], tube_radius=3, color=(255/255, 181/255, 187/255))
mlab.plot3d(UAV_W_RL[0, 0:max_steps], UAV_W_RL[1, 0:max_steps], ss1[0:max_steps], tube_radius=3, color=(157/255, 181/255, 187/255))
mlab.plot3d(FOV_C_RL[0:max_step_size, 0], FOV_C_RL[0:max_step_size, 1], ss1[0:max_steps], tube_radius=3, color=(125/255, 164/255, 208/255))
mlab.plot3d(FOV_C_W_RL[0:max_step_size, 0], FOV_C_W_RL[0:max_step_size, 1] , ss1[0:max_steps], tube_radius=3, color=(34/255, 164/255, 208/255))
mlab.plot3d(x_e_rl[0:100, 0], y_e_rl[0:100, 0] , ss1[0:100], tube_radius=3, color=(0,0,0))
mlab.plot3d(x_e_w_rl[0:100, 0], y_e_w_rl[0:100, 0] , ss1[0:100], tube_radius=3, color=(1,1,1))
mlab.mesh(Xc_1, Yc_1, Zc_1)
mlab.mesh(Xc_2, Yc_2, Zc_2)
mlab.mesh(Xc_3, Yc_3, Zc_3)
mlab.title('Tracking UAV')
s = [0, 1900, -300, 1000, 0, 170]
mlab.orientation_axes(xlabel='X-axis',ylabel='Y-axis',zlabel='Z-axis')
mlab.outline(color=(0, 0, 0), extent=s)
mlab.axes(color=(0, 0, 0), extent=s)

mlab.show()
plt.show()
