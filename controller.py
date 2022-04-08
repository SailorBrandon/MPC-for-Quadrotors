from re import I
from terminal_set import Terminal_set
import numpy as np
from scipy.spatial.transform import Rotation
import quadrotor
import cvxpy as cp
import trajectory
import numpy.linalg as LA
import control
import math
import matplotlib.pyplot as plt
# from nonlinear_mpc_solver import *
import scipy.io


class Controller:
    def __init__(self, traj, ctrl_freq):
        self.quad_model = quadrotor.Quadrotor()
        self.traj = traj
        self.ctrl_freq = ctrl_freq
        self.mass = self.quad_model.mass
        self.g = self.quad_model.g
        self.arm_length = self.quad_model.arm_length
        self.rotor_speed_min = 0
        self.rotor_speed_max = 2500
        self.k_thrust = self.quad_model.k_thrust
        self.k_drag = self.quad_model.k_drag
        self.to_TM = self.quad_model.to_TM
        self.inertia = np.diag(
            [self.quad_model.Ixx, self.quad_model.Iyy, self.quad_model.Izz])

    def generate_control_input(self, u):
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        # cmd_q = np.zeros((4,))

        F = (np.linalg.inv(self.to_TM) @ u).astype(float)

        for i in range(4):
            if F[i] < 0:
                F[i] = 0
                cmd_motor_speeds[i] = self.rotor_speed_min
            cmd_motor_speeds[i] = np.sqrt(F[i] / self.k_thrust)
            if cmd_motor_speeds[i] > self.rotor_speed_max:
                cmd_motor_speeds[i] = self.rotor_speed_max

        cmd_thrust = u[0]  # thrust
        cmd_moment[0] = u[1]  # moment about p
        cmd_moment[1] = u[2]  # moment about q
        cmd_moment[2] = u[3]  # moment about r

        control_input = {'cmd_motor_speeds': cmd_motor_speeds,
                         'cmd_thrust': cmd_thrust,
                         'cmd_moment': cmd_moment}
        return control_input
    
    def state2x(self, state):
        x = state.get('x').flatten()
        v = state.get('v').flatten()
        try:
            q = state.get('q').flatten()
            w = state.get('w').flatten()
            euler_ang = self.euler_from_quaternion(q[0], q[1], q[2], q[3])
        except:
            euler_ang = np.zeros(3)
            euler_ang[2] = state.get('yaw')
            w = np.zeros(3)
            w[2] = state.get('yaw_dot')

        x_init = np.block([x, v, euler_ang, w])
        return x_init

    def euler_from_quaternion(self, x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return np.array([roll_x, pitch_y, yaw_z])  # in radians


class PDcontroller(Controller):
    def __init__(self, traj, ctrl_freq):
        Controller.__init__(self, traj, ctrl_freq)
        # hover control gains
        self.Kp = np.diag([10, 10, 200])
        self.Kd = np.diag([10, 10, 3])
        # angular control gains
        self.Kp_t = np.diag([250, 250, 30])
        self.Kd_t = np.diag([30, 30, 7.55])

    def control(self, cur_time, obs_state):
        '''
        :param desired state: pos, vel, acc, yaw, yaw_dot
        :param current state: pos, vel, euler, omega
        :return:
        '''
        des_state = self.traj.get_des_state(cur_time)
        # position controller
        error_pos = des_state.get('x') - obs_state.get('x').reshape(3, 1)
        error_vel = des_state.get('v') - obs_state.get('v').reshape(3, 1)
        rdd_des = des_state.get('x_ddt')
        rdd_cmd = rdd_des + self.Kd @ error_vel + self.Kp @ error_pos
        u1 = self.mass * (self.g + rdd_cmd[2]).reshape(-1, 1)

        # attitude controller
        psi_des = des_state.get('yaw')
        phi_cmd = (rdd_cmd[0] * np.sin(psi_des) -
                   rdd_cmd[1] * np.cos(psi_des)) / self.g
        theta_cmd = (rdd_cmd[0] * np.cos(psi_des) +
                     rdd_cmd[1] * np.sin(psi_des)) / self.g
        quat = obs_state['q']
        rotation = Rotation.from_quat(quat)
        angle = np.array(rotation.as_rotvec()).reshape(3, 1)  # euler angles
        omega = np.array(obs_state['w']).reshape(3, 1)  # anglar velocity
        psid_des = des_state.get('yaw_dot')
        ang_ddt = self.Kd_t @ (np.array([[0], [0], [psid_des]]) - omega) + \
            self.Kp_t @ (np.array([[phi_cmd[0]],
                         [theta_cmd[0]], [psi_des]]) - angle)
        u2 = self.inertia @ ang_ddt

        u = np.vstack((u1, u2))
        print("control input: ", u)
        control_input = self.generate_control_input(u)
        return control_input, error_pos


class NonLinear_MPC(Controller):
    def __init__(self, traj, ctrl_freq):
        super().__init__(traj, ctrl_freq)
        self.param = MPC_Formulation_Param()
        self.param.set_horizon(dt=1/self.ctrl_freq, N=10)
        self.solver = acados_mpc_solver_generation(
            self.param, collision_avoidance=False)

    def control(self, cur_time, obs_state):
        des_state = self.traj.get_des_state(cur_time)
        error_pos = des_state.get('x') - obs_state.get('x').reshape(3, 1)

        x_init = self.state2x(obs_state)
        self.solver.set(0, "lbx", x_init)
        self.solver.set(0, "ubx", x_init)

        mpc_time = cur_time
        for k in range(self.param.N):
            mpc_time += k * self.param.dt
            des_state_ahead = self.traj.get_des_state(mpc_time)
            x_ref_k = self.state2x(des_state_ahead)
            u_ref_k = np.array([self.g, 0, 0, 0])
            yref = np.block([x_ref_k, u_ref_k])
            self.solver.set(k, "yref", yref)

        yref_e = yref[:6]
        self.solver.set(self.param.N, 'yref', yref_e)

        status = self.solver.solve()
        if status != 0:
            print("acados returned status {} in closed loop iteration {}.".format(
                status, cur_time))

        u = self.solver.get(0, "u")
        u[0] *= self.mass
        x = self.solver.get(0, "x")
        control_input = self.generate_control_input(u)
        return control_input, error_pos


class Linear_MPC(Controller):
    def __init__(self, traj, ctrl_freq):
        Controller.__init__(self, traj, ctrl_freq)
        self.ctrl_freq = ctrl_freq
        self.dt = 1 / self.ctrl_freq
        self.Ad, self.Bd = self.quad_model.get_dLTI(self.dt)
        self.B_dist = np.block([[np.zeros((3, 3))],
                                [np.eye(3)],
                                [np.zeros((3, 3))],
                                [np.zeros((3, 3))]])
        
        self.disturbance_observer = Luenberger_Observer(self.Ad, self.Bd, self.B_dist, C=np.eye(12), C_dist=np.zeros((12, 3)), load_L=False)
        
        C_obs = np.zeros((6, 12))
        C_obs[:3, :3] = np.eye(3)
        C_obs[3:, 6:9] = np.eye(3)
        self.vel_observer = Luenberger_Observer(self.Ad, self.Bd, self.B_dist, C=C_obs, C_dist=np.zeros((6, 3)), load_L=True)
        
        self.N = 10  # the number of predicted steps TODO
        # C = control.ctrb(self.Ad, self.Bd) # rank(C)=12, controllable
        # subQ_pos = np.block([[1.2e5*np.eye(3), np.zeros((3, 3))],
        #                      [np.zeros((3, 3)), 8e2*np.eye(3)]])
        # subQ_ang = np.block([[5e3*np.eye(3), np.zeros((3, 3))],
        #                      [np.zeros((3, 3)), 0e2*np.eye(3)]])
        # self.Q = np.block([[subQ_pos, np.zeros((6, 6))],
        #                    [np.zeros((6, 6)), subQ_ang]])
        subQ_pos = np.block([[1e3*np.eye(3), np.zeros((3, 3))],
                             [np.zeros((3, 3)), 1e2*np.eye(3)]])
        subQ_ang = np.block([[1e3*np.eye(3), np.zeros((3, 3))],
                             [np.zeros((3, 3)), 1e2*np.eye(3)]])
        self.Q = np.block([[subQ_pos, np.zeros((6, 6))],
                           [np.zeros((6, 6)), subQ_ang]])
        # self.Q = 1e4*np.eye(12)
        self.R = 1e1*np.eye(4)
        self.P, self.K = self.get_terminal_cost(
            self.Ad, self.Bd, self.Q, self.R)

        self.Ak = self.Ad-self.Bd@self.K
        self.Hx = np.array([
                            [0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0],  # roll pitch constraints
                            [0, 0, 0, 0, 0, 0, -1.0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], #velocity constraints
                            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, -1.0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0],
                            [0, 0, 0, -1.0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, -1.0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, -1.0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, -1.0, 0, 0, 0],
                            ])
        self.Hu = np.array([[1/self.mass, 0, 0, 0],
                            [-1/self.mass, 0, 0, 0]
                            ])  # z acc constraints
        self.Hu1 = np.array([[1/self.mass, 0, 0, 0],
                            [-1/self.mass, 0, 0, 0],
                            [0, 1.0, 0, 0],
                            [0, -1.0, 0, 0],
                            [0, 0, 1.0, 0],
                            [0, 0, -1.0, 0],
                            [0, 0, 0, 1.0],
                            [0, 0, 0, -1.0]
                            ])
        self.h = np.array([[0.5*self.g],  # z acc constraints
                           [0.5*self.g],
                           [0.5],  
                           [0.5],
                           [2.0],#velocity constraints
                           [2.0],
                           [2.0],
                           [0.5], 
                           [0.5],
                           [2.0],
                           [2.0],
                           [2.0],
                           [0.5],  
                           [0.5]
                           ])
        self.h1 = np.array([[1.5*self.g],  # z acc constraints
                           [-0.5*self.g],
                           [0.15],  
                           [0.15],
                           [0.15],
                           [0.15],  
                           [0.15],  
                           [0.15],
                           [0.5],  
                           [0.5],
                           [2.0],#velocity constraints
                           [2.0],
                           [2.0],
                           [0.5], 
                           [0.5],
                           [2.0],
                           [2.0],
                           [2.0],
                           [0.5],  
                           [0.5]
                           ])
        self.terminal_set = Terminal_set(
            self.Hx, self.Hu, self.K, self.Ak, self.h)
        self.Xf_nr = self.terminal_set.Xf_nr
        self.terminal_set.test_input_inbound(u_limit=0.15)
        # raise NotImplementedError()
        self.x_real = [[], [], []]
        self.x_obsv = [[], [], []]

    def control(self, cur_time, obs_state):
        des_state = self.traj.get_des_state(cur_time)
        error_pos = des_state.get('x') - obs_state.get('x').reshape(3, 1)

        x_sys = self.state2x(obs_state)
        x_obs_dist = self.disturbance_observer.x_hat.flatten()
        x_obs_vel = self.vel_observer.x_hat.flatten()
        y = np.block([x_sys[:3], x_sys[6:9]])
        x_init = x_sys
        
        for i in range(3):
            self.x_real[i].append(x_sys[i])
            self.x_obsv[i].append(x_obs_vel[i])

        x = cp.Variable((12, self.N+1))
        u = cp.Variable((4, self.N))
        cost = 0
        constr = []
        mpc_time = cur_time
        desired_x = []
        d_hat = self.disturbance_observer.d_hat
        for k in range(self.N+1):
            mpc_time += k * self.dt
            des_state_ahead = self.traj.get_des_state(mpc_time)
            x_ref_k = self.state2x(des_state_ahead)
            desired_x.append(x_ref_k)
            if k == self.N:
                cost += cp.quad_form(x[:, self.N]-x_ref_k, self.P)
                constr.append(
                    self.Xf_nr[0] @ (x[:, self.N]-x_ref_k) <= self.Xf_nr[1].squeeze())
                break
            cost += cp.quad_form(x[:, k] - x_ref_k, self.Q)
            u_ref_k = np.array([self.mass*self.g, 0, 0, 0])
            cost += cp.quad_form(u[:, k] - u_ref_k, self.R)

            constr.append(self.Hx @ x[:, k] <= self.h1[self.Hu1.shape[0]:].squeeze())
            constr.append(self.Hu1 @ u[:, k] <= self.h1[:self.Hu1.shape[0]].squeeze())
            gravity = np.zeros([12, ])
            gravity[5] = self.dt*self.g  # discritized
            constr.append(x[:, k + 1] == self.Ad @ x[:, k] +
                          self.Bd @ u[:, k] - gravity + (self.B_dist@d_hat).flatten())

        constr.append(x[:, 0] == x_init)
        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve(verbose=False)
        u = u[:, 0].value
        print(u)

        
        # self.disturbance_observer.update(u, x_sys)
        # self.vel_observer.update(u, y)
        
        control_input = self.generate_control_input(u)
        return control_input, error_pos

    def get_terminal_cost(self, Ad, Bd, Q, R):
        P, L, G = control.dare(Ad, Bd, Q, R)
        return P, G


class Luenberger_Observer:
    def __init__(self, A, B, B_dist, C, C_dist, load_L=False):
        self.x_hat = np.zeros((12, 1))
        self.d_hat = np.zeros((3, 1))
        self.A_sq = np.block([[A, B_dist],
                              [np.zeros((3, 12)), np.eye(3)]])
        self.B_sq = np.block([[B],
                              [np.zeros((3, 4))]])
        self.C_sq = np.block([C, C_dist])
        if load_L:
            file_path = 'saveL.mat'
            mat = scipy.io.loadmat(file_path)
            self.L = mat['L']
        else:
            eig = [0.5 for i in range(7)] + [0.7 for i in range(8)]
            self.L = control.place(self.A_sq.T, self.C_sq.T, eig).T

    def update(self, u, y):
        exd_state = np.block([[self.x_hat],
                              [self.d_hat]])
        exd_state = self.A_sq@exd_state + \
            self.B_sq@u.reshape(-1, 1) + \
            self.L@(y.reshape(-1, 1) - self.C_sq@exd_state)
        self.x_hat = exd_state[:12]
        self.d_hat = exd_state[12:]

