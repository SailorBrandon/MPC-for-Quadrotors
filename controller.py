from re import I
import numpy as np
from scipy.spatial.transform import Rotation
import quadrotor
import cvxpy as cp
import trajectory
import numpy.linalg as LA
import control
import math
import matplotlib.pyplot as plt


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

        cmd_thrust = u[0]  # thruse
        cmd_moment[0] = u[1]  # moment about p
        cmd_moment[1] = u[2]  # moment about q
        cmd_moment[2] = u[3]  # moment about r

        control_input = {'cmd_motor_speeds': cmd_motor_speeds,
                         'cmd_thrust': cmd_thrust,
                         'cmd_moment': cmd_moment}
        return control_input


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
        # print("control input: ", u)
        control_input = self.generate_control_input(u)
        return control_input, error_pos


class Linear_MPC(Controller):
    def __init__(self, traj, ctrl_freq):
        Controller.__init__(self, traj, ctrl_freq)
        self.ctrl_freq = ctrl_freq
        self.dt = 1 / self.ctrl_freq

    def control(self, cur_time, obs_state):
        des_state = self.traj.get_des_state(cur_time)
        error_pos = des_state.get('x') - obs_state.get('x').reshape(3, 1)

        x_init = self.state2x(obs_state)
        N = 5  # the number of predicted steps
        x = cp.Variable((12, N+1))
        u = cp.Variable((4, N))
        cost = 0
        constr = []
        # subQ_pos = np.block([[1e4*np.eye(3), np.zeros((3, 3))],
        #                      [np.zeros((3, 3)), 5e1*np.eye(3)]])
        # subQ_ang = np.block([[1.5e4*np.eye(3), np.zeros((3, 3))],
        #                      [np.zeros((3, 3)), 0e0*np.eye(3)]])
        # Q = np.block([[subQ_pos, np.zeros((6, 6))],
        #               [np.zeros((6, 6)), subQ_ang]])
        # R = 1e2*np.eye(4)

        subQ_pos = np.block([[1.2e4*np.eye(3), np.zeros((3, 3))],
                             [np.zeros((3, 3)), 1e3*np.eye(3)]])
        subQ_ang = np.block([[5e2*np.eye(3), np.zeros((3, 3))],
                             [np.zeros((3, 3)), 0e2*np.eye(3)]])
        # subQ_pos[2,2]=2e5
        Q = np.block([[subQ_pos, np.zeros((6, 6))],
                      [np.zeros((6, 6)), subQ_ang]])
        R = 0e1*np.eye(4)
        mpc_time = cur_time
        desired_x=[]
        for k in range(N):
            mpc_time += (k+1) * self.dt
            des_state_ahead = self.traj.get_des_state(mpc_time)
            # print(des_state_ahead['yaw'],des_state_ahead['yaw_dot'])
            desired_x.append(_pack_state(des_state_ahead))
            x_ref_k = self.state2x(des_state_ahead)
            cost += cp.quad_form(x[:, k+1] - x_ref_k, Q)
            # u_ref_k = np.array([self.mass*self.g, 0, 0, 0])
            # cost += cp.quad_form(u[:, k] - u_ref_k, R)
            # constr.append(cp.norm(x[7:9, k+1],'inf') <= 0.5)
            cost += cp.quad_form(u[:, k], R)
            Ad, Bd = self.get_LTV(des_state_ahead,x_init,k)
            constr.append(x[:, k + 1] == Ad @ x[:, k] + Bd @ u[:, k])

        constr.append(x[:, 0] == x_init)
        # constr.append(x[:, N-1] == x_ref_k)
        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve(solver=cp.OSQP,verbose=True,max_iter=40000)
        u = u[:, 0].value
        optimized_x=(x[:, 1:].value).T
        # if ((int(cur_time*100)/100)%0.5 < 0.01) and cur_time>2.8:
        #     print(cur_time)
        #     visualize_x(np.array(desired_x),optimized_x,self.dt)
        #     print(self.traj.get_des_state(mpc_time))
        # print("control input: ", u)
        control_input = self.generate_control_input(u)
        return control_input, error_pos

    def get_LTV(self, des_state_ahead,x_init,k):
        num_x = 12
        num_u = 4
        x7=x_init[6]
        x8=x_init[7]
        x9_bar = des_state_ahead.get('yaw')
        sinx9=np.sin(x9_bar)
        cosx9=np.cos(x9_bar)
        x6_dot_bar = des_state_ahead.get('x_ddt')[2]
        u1_bar = x6_dot_bar + self.g   # =  real_u1_bar/self.mass
        # Linearized state space model
        Ixx = self.quad_model.Ixx
        Iyy = self.quad_model.Iyy
        Izz = self.quad_model.Izz
        # print(x9_bar)
        Ac = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, (u1_bar*sinx9)[0], (u1_bar*cosx9)[0], 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, (-u1_bar*cosx9)[0], (u1_bar*sinx9)[0], 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0, 0, 0, (-Iyy+Izz)*x9_bar/Ixx, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, (-Izz+Ixx)*x9_bar/Iyy, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ])
        Bc = np.array([[0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [1/self.mass, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 1/self.quad_model.Ixx, 0, 0],
                       [0, 0, 1/self.quad_model.Iyy, 0],
                       [0, 0, 0, 1/self.quad_model.Izz]])
        # if k==0:
        #     Ac[5,6] =-(u1_bar*np.sin(x7)*np.cos(x8))[0]
        #     Ac[5,7] =-(u1_bar*np.sin(x8)*np.cos(x7))[0]
        Cc = np.eye(num_x)
        Dc = np.zeros((num_x, 4))
        sysc = control.ss(Ac, Bc, Cc, Dc)
        # Discretization
        sysd = control.sample_system(sysc, self.dt, method='bilinear')
        return sysd.A, sysd.B

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
    
def visualize_x(desired_x,optimized_x,dt):
    t=np.arange(desired_x.shape[0])*dt
    i=6
    plt.subplot(2, 3, 1)
    plt.plot(t,desired_x[:,i])
    plt.plot(t,optimized_x[:,i])
    plt.xlabel('t')
    plt.ylabel('state value')
    plt.title('x')
    plt.legend(['desired', 'optimized'])
    plt.subplot(2, 3, 2)
    plt.plot(t, desired_x[:, i+1])
    plt.plot(t, optimized_x[:, i+1])
    plt.xlabel('t')
    plt.ylabel('state value')
    plt.title('y')
    plt.legend(['desired', 'optimized'])
    plt.subplot(2, 3, 3)
    plt.plot(t, desired_x[:, i+2])
    plt.plot(t, optimized_x[:,i+2])
    plt.xlabel('t')
    plt.ylabel('state value')
    plt.title('z')
    plt.legend(['desired', 'optimized'])
    plt.subplot(2, 3, 4)
    plt.plot(t, desired_x[:, i+3])
    plt.plot(t, optimized_x[:, i+3])
    plt.xlabel('t')
    plt.ylabel('state value')
    plt.title('vx')
    plt.legend(['desired', 'optimized'])
    plt.subplot(2, 3, 5)
    plt.plot(t, desired_x[:, i+4])
    plt.plot(t, optimized_x[:, i+4])
    plt.xlabel('t')
    plt.ylabel('state value')
    plt.title('vy')
    plt.legend(['desired', 'optimized'])
    plt.subplot(2, 3, 6)
    plt.plot(t, desired_x[:, i+5])
    plt.plot(t, optimized_x[:, i+5])
    plt.xlabel('t')
    plt.ylabel('state value')
    plt.title('vz')
    plt.legend(['desired', 'optimized'])
    plt.show()

def _pack_state(state):
    """
    Convert a state dict to Quadrotor's private internal vector representation.
    """
    s = np.zeros((12,))
    s[0:3] = state['x'].squeeze()
    s[3:6] = state['v'].squeeze()
    s[8] = state['yaw']
    s[11] = state['yaw_dot']
    return s