from gym import Env
from gym.spaces import Discrete, Box, Tuple
import numpy as np
import random
import gym
from time import time
import matplotlib.pyplot as plt
import casadi as ca
from casadi import sin, cos, pi, tan

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
x0 = [97, 150, 80, 0, 0, 0, 0, 0]  # initial UAV states
xs = [100, 150, 0]  # initial target state
mpc_iter = 0  # initial MPC count
max_step_size = 500
ss1 = np.zeros((701))

# function to shift MPC one step ahead
def shift_timestep(T, t0, x0, u, f_u, xs):
    f_value = f_u(x0, u[:, 0])
    x0 = ca.DM.full(x0 + (T * f_value))

    t0 = t0 + T
    u0 = ca.horzcat(
        u[:, 1:],
        ca.reshape(u[:, -1], -1, 1)
    )

    con_t = [12, 0.045]  # Linear and angular velocity of target
    f_t_value = ca.vertcat(con_t[0] * cos(xs[2]),
                           con_t[0] * sin(xs[2]),
                           con_t[1])
    xs = ca.DM.full(xs + (T * f_t_value))
    return t0, x0, u0, xs


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
    x_o_1 = 175
    y_o_1 = 779
    x_o_2 = -134
    y_o_2 = 155
    x_o_3 = 200
    y_o_3 = 343
    obs_r = 30
    UAV_r = 5

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
                       -ca.sqrt((X[0, k] - x_o_3) ** 2 + (X[1, k] - y_o_3) ** 2) + (UAV_r + obs_r)
                       # limit of obstacle-3
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
    lbg = ca.DM.zeros((n_states_u * (N + 1)))
    ubg = ca.DM.zeros((n_states_u * (N + 1)))

    # Constrains on states (Inequality constrains)
    lbg[0:128:8] = z_u_min  # z lower bound
    lbg[1:128:8] = theta_u_min  # theta lower bound
    lbg[2:128:8] = phi_g_min  # phi lower bound
    lbg[3:128:8] = theta_g_min  # theta lower bound
    lbg[4:128:8] = shi_g_min  # shi lower bound
    lbg[6:128:8] = -ca.inf  # Obstacle - 1
    lbg[5:128:8] = -ca.inf  # Obstacle - 2
    lbg[7:128:8] = -ca.inf  # Obstacle - 3

    ubg[0:128:8] = z_u_max  # z lower bound
    ubg[1:128:8] = theta_u_max  # theta lower bound
    ubg[2:128:8] = phi_g_max  # phi lower bound
    ubg[3:128:8] = theta_g_max  # theta lower bound
    ubg[4:128:8] = shi_g_max  # shi lower bound
    ubg[6:128:8] = 0  # Obstacle - 1
    ubg[5:128:8] = 0  # Obstacle - 2
    ubg[7:128:8] = 0  # Obstacle - 3

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

    global x0, xs, mpc_iter
    t0 = 0

    # xx = DM(state_init)
    t = ca.DM(t0)

    u0 = ca.DM.zeros((n_controls, N))  # initial control
    xx = ca.DM.zeros((8, 801))  # change size according to main loop run, works as tracker for target and UAV
    #ss = ca.DM.zeros((3, 801))
    x_e_1 = ca.DM.zeros((801))
    y_e_1 = ca.DM.zeros((801))

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

            a_p = (x0[2] * (tan(x0[6] + VFOV / 2)) - x0[2] * tan(x0[6] - VFOV / 2)) / 2  # For plotting FOV
            b_p = (x0[2] * (tan(x0[5] + HFOV / 2)) - x0[2] * tan(x0[5] - HFOV / 2)) / 2
            x_e_1[mpc_iter] = x0[0] + a_p + x0[2] * (tan(x0[6] - VFOV / 2))
            y_e_1[mpc_iter] = x0[1] + b_p + x0[2] * (tan(x0[5] - HFOV / 2))

            FOV_X = x_e_1[mpc_iter]
            FOV_Y = y_e_1[mpc_iter]

            error = ca.DM.zeros(701)

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
        if (action[0] == 0 or action[1] == 0):
            error, obs = MPC(1, 1)
            print("It's 0, -2 reward")
            reward = -2
        else:
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
        global x0, xs, mpc_iter, max_step_size, ss
        x0 = [97, 150, 80, 0, 0, 0, 0, 0]
        xs = [100, 150, 0]
        mpc_iter = 0
        ss = ca.DM.zeros((3, 801))
        self.episode_length = max_step_size


env = Tunning()

######################### Testing Evn #########################
"""
episodes = 5
for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        # env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))

"""
#################################################################
########################### Q-learning ##########################
step_size = max_step_size+1  # Change according to main loop run

qtable = np.zeros((step_size, 101, 101))
"""
print(np.shape(qtable))
qtable[5, 10, 10] = 50
qtable[5, 5, 5] = 15
qtable[5, 1, 10] = 200
print(qtable[5, :, :])
#print(np.shape(qtable[5, :, :]))  # gives tuple
print(max_index(qtable[5, :, :]))

#print(np.argmax(qtable[5, :, :]))
#qtable[2, 2, 3] = 5
#print(qtable)  np.argmax(qtable[2,:,:])
#test = 91
#print((test//10, test%10))
"""

# Q - learning parameters
total_episodes = 1000  # Total episodes
learning_rate =  1 # Learning rate 0.8 is good
max_steps = max_step_size  # Max steps per episode
gamma = 0.1  # Discounting rate 0.1 is good

# Exploration parameters
epsilon = 1.0  # Exploration rate
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.03  # Minimum exploration probability
decay_rate = 0.0045   # Exponential decay rate for exploration prob

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
UAV_RL = ca.DM.zeros((3,801))
UAV_W_RL = ca.DM.zeros((3,801))
targetarr = ca.DM.zeros((3,801))
error_w_rl = np.zeros((max_steps))

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
    #global x0, xs
    print('\n-------- We are in episode {}, {} steps will be run --------\n'.format(episode + 1, max_steps))

    env.reset()
    step = 0
    done = False
    total_rewards = 0
    #print(qtable)

    while not done:
        # 3. Choose an action a in the current world state (s)
        ## First we randomize a number
        exp_exp_tradeoff = random.uniform(0, 1)

        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            action = max_index(qtable[step, :, :])
            #print("Exploit")

        # Else doing a random choice --> exploration
        else:
            action = (0,0)
            while(action[0] == 0 or action[1] == 0):
                action = env.action_space.sample()

        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, obs2, reward, done, info, error, a_p, b_p, x_e_1, y_e_1 = env.step(action)


        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        # qtable[new_state,:] : all the actions we can take from new state
        qtable[step, action[0], action[1]] = (1-learning_rate)*(qtable[step, action[0], action[1]]) + \
                                                 learning_rate * (reward + gamma * np.max(qtable[step+1, :, :]))

        print("step is {}".format(step))
        #print(qtable[step, :, :])
        #print(action[0],action[1])

        step += 1
        total_rewards += reward
        rewardarr[episode, step - 1] = reward
        errorarr[episode, step - 1] = error

        w_1[step - 1, episode] = action[0]
        w_2[step - 1, episode] = action[1]

        if (episode == (total_episodes - 1)):
            action_str[step, 0] = action[0]
            action_str[step, 1] = action[1]

    erroravg[episode] = (sum(errorarr[episode,:])/max_step_size)
    rewardavg[episode] = (sum(rewardarr[episode,:])/max_step_size)

    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    rewards.append(total_rewards)



Error = 0
env.reset()
UAV_RL[:, 0] = x0[0:3]
targetarr[:, 0] = xs
#Printing Optimal Policy
for i in range(max_step_size):
    action = max_index(qtable[i, :, :])
    new_state, obs2, reward, done, info, error, a_p, b_p, x_e_1, y_e_1 = env.step(action)
    UAV_RL[:, i+1] = new_state
    targetarr[:, i+1] = obs2
    maxreward[i] = reward
    minerrorarr[i] = error
    Error += error
    print(max_index(qtable[i, :, :]))

x_e_rl, y_e_rl = ellipse(a_p, b_p, x_e_1, y_e_1)
x_e_rl = np.array(x_e_rl)
y_e_rl = np.array(y_e_rl)
print('Error of Optimal Policy: {}'.format(Error))

# for without RL error and trajectory
Error = 0
env.reset()
UAV_W_RL[:, 0] = x0[0:3]
#Printing Optimal Policy
for i in range(max_step_size):
    action = (50, 1)
    new_state, obs2, reward, done, info, error, a_p, b_p, x_e_1, y_e_1 = env.step(action)
    UAV_W_RL[:, i+1] = new_state
    error_w_rl[i] = error
    Error += error
    #print(max_index(qtable[i, :, :]))
x_e_w_rl, y_e_w_rl = ellipse(a_p, b_p, x_e_1, y_e_1)
x_e_w_rl = np.array(x_e_w_rl)
y_e_w_rl = np.array(y_e_w_rl)
print('Error without RL: {}'.format(Error))

################################################################
################### Saving all required arrays #################

np.savez("1k_1_01_500.npz", errorarr=errorarr, w_1=w_1, w_2=w_2, erroravg=erroravg, rewardavg=rewardavg, rewardarr=rewardarr, qtable=qtable)


################################################################
############# Plotting the reward & printing actions ###########
last_action = action_str[1:51, 0:2]
sum_without_rl = sum(error_w_rl)
sum_first_episode = sum(errorarr[0, 0:max_steps])
sum_last_episode = sum(errorarr[total_episodes-1, 0:max_steps])
sum_optimal_policy = sum(minerrorarr)
x_axis_bar = ['Error without RL', 'Error of FirstEpisode', 'Error of LastEpisode', 'Error of OptimalPolicy']
y_axis_bar = [sum_without_rl, sum_first_episode, sum_last_episode, sum_optimal_policy]

# Printing actions
#print(last_action)
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
plt.plot(rewardarr[total_episodes//2, 0:max_steps], color="green")
plt.plot(rewardarr[total_episodes-1, 0:max_steps], color="brown")
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

#plotting error
fig = plt.figure()
plt.plot(erroravg, color="blue")
plt.title('Sum of Error')  # Title of the plot
plt.xlabel('Episode')  # X-Label
plt.ylabel('Sum of Error')  # Y-Label

fig = plt.figure()
plt.subplot(121)
plt.plot(errorarr[0, 0:max_steps], color="red")
plt.plot(minerrorarr[0:max_steps], color="blue")
plt.plot(error_w_rl[0:max_steps], color="brown")
plt.legend(loc=4)
plt.legend(['First Episode Error', 'Error of Optimal Policy', 'Error without RL'])
plt.title('Error Plot')  # Title of the plot
plt.xlabel('Steps')  # X-Label
plt.ylabel('Error')  # Y-Label
plt.subplot(122)
plt.bar(x_axis_bar, y_axis_bar)

#plottting UAV and target with and without RL
UAV_RL = np.array(UAV_RL)
UAV_W_RL = np.array(UAV_W_RL)
targetarr = np.array(targetarr)

fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot3D(UAV_RL[0, 0:max_steps], UAV_RL[1, 0:max_steps], UAV_RL[2, 0:max_steps], linewidth = "2", color = "brown")
ax.plot3D(UAV_W_RL[0, 0:max_steps], UAV_W_RL[1, 0:max_steps], UAV_W_RL[2, 0:max_steps], linewidth = "2", color = "blue")
ax.plot3D(targetarr[0,0:max_steps], targetarr[1,0:max_steps], ss1[0:max_steps], '--',linewidth = "2", color = "red")
ax.plot3D(UAV_RL[0, 0:max_steps], UAV_RL[1, 0:max_steps], ss1[0:max_steps], linewidth = "2", color = "pink")
ax.plot3D(UAV_W_RL[0, 0:max_steps], UAV_W_RL[1, 0:max_steps], ss1[0:max_steps], linewidth = "2", color = "green")
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
ax.legend(['UAV With RL', 'UAV Without RL', 'Target', 'UAV With RL Projection', 'UAV Without RL Projection', 'RL FOV', 'Without RL FOV'])

plt.show()
