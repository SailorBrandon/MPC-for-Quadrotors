import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

class Trajectory:
    def __init__(self, input_traj="hover"):
        self.heading = np.zeros((2))
        self.yaw = 0
        if input_traj == "diamond":
            self.input_traj = self.diamond
        elif input_traj == "oneline":
            self.input_traj = self.oneline
        elif input_traj == "hover":
            self.input_traj = self.hover
        elif input_traj == "circle":
            self.input_traj = self.circle
    def circle(self,t):
        T = 14
        radius = 3
        dt = 0.01
        if t > T:
            pos = np.array([radius, 0, 2.5])
            vel = np.array([0,0,0])
            acc = np.array([0,0,0])
        else:
            angle,_,_ = self.tj_from_line(0, 2*np.pi, T, t)
            angle2,_,_ = self.tj_from_line(0, 2*np.pi, T, t+dt)
            angle3,_,_ = self.tj_from_line(0, 2*np.pi, T, t+2*dt)
            pos = np.array([radius*(np.cos(angle)-1),radius*np.sin(angle),2.5*angle/(2*np.pi)])
            pos2 = np.array([radius*(np.cos(angle2)-1),radius*np.sin(angle2),2.5*angle2/(2*np.pi)])
            pos3 = np.array([radius*(np.cos(angle3)-1),radius*np.sin(angle3),2.5*angle3/(2*np.pi)])
            vel = (pos2- pos)/dt
            vel2 = (pos3- pos2)/dt
            acc = (vel2 - vel)/dt
            pos=pos.reshape([-1,1])
            vel=vel.reshape([-1,1])
            acc=acc.reshape([-1,1])
        return pos, vel, acc
    def diamond(self, t):
        T1, T2, T3, T4 = 3.5, 3.5, 3.5, 3.5
        points = []
        points.append(np.zeros((3, 1)))
        points.append(np.array([[0], [np.sqrt(2)], [np.sqrt(2)]]))
        points.append(np.array([[1], [0], [2*np.sqrt(2)]]))
        points.append(np.array([[1], [-np.sqrt(2)], [np.sqrt(2)]]))
        points.append(np.array([[1], [0], [0]]))
        if (0 < t) and (t <= T1):
            pos, vel, acc = self.tj_from_line(points[0], points[1], T1, t)
        elif (T1 < t) and (t <= (T1+T2)):
            pos, vel, acc = self.tj_from_line(points[1], points[2], T2, t-T1)
        elif ((T1 + T2) < t) and (t <= (T1 + T2 + T3)):
            pos, vel, acc = self.tj_from_line(
                points[2], points[3], T3, t - (T1 + T2))
        elif ((T1 + T2 + T3) < t) and (t <= (T1 + T2 + T3 + T4)):
            pos, vel, acc = self.tj_from_line(
                points[3], points[4], T4, t - (T1 + T2 + T3))
        elif (t > (T1 + T2 + T3 + T4)):
            pos, vel, acc = points[4], np.zeros((3, 1)), np.zeros((3, 1))
        else:
            pos, vel, acc = np.zeros((3, 1)), np.zeros(
                (3, 1)), np.zeros((3, 1))
        return pos, vel, acc

    def oneline(self, t):
        T1 = 5.0
        points = []
        points.append(np.zeros((3, 1)))
        points.append(np.array([[0], [1], [1]]))
        if (0 < t) and (t <= T1):
            pos, vel, acc = self.tj_from_line(points[0], points[1], T1, t)
        else:
            pos, vel, acc = points[-1], np.zeros((3, 1)), np.zeros((3, 1))
        return pos, vel, acc

    def hover(self, t):
        pos, vel, acc = np.zeros((3, 1)), np.zeros((3, 1)), np.zeros((3, 1))
        return pos, vel, acc

    def tj_from_line(self, start_pos, end_pos, time_ttl, t_c):
        v_max = (end_pos - start_pos) * 2 / time_ttl
        if (t_c >= 0 and t_c < time_ttl/2):
            vel = v_max*t_c/(time_ttl/2)
            pos = start_pos + t_c*vel/2
            acc = v_max/(time_ttl/2)
        elif (t_c >= time_ttl/2 and t_c <= time_ttl):
            vel = v_max * (time_ttl - t_c) / (time_ttl / 2)
            pos = end_pos - (time_ttl - t_c) * vel / 2
            acc = - v_max/(time_ttl / 2)
        else:
            if (type(start_pos) == int) or (type(start_pos) == float):
                pos, vel, acc = 0, 0, 0
            else:
                pos, vel, acc = np.zeros(start_pos.shape), np.zeros(
                    start_pos.shape), np.zeros(start_pos.shape)
        return pos, vel, acc

    def get_yaw(self, vel):
        if np.allclose(vel, np.zeros((3, 1))):
            vel += 1e-5
        curr_heading = vel/LA.norm(vel)
        prev_heading = self.heading
        cosine = max(-1, min(np.dot(prev_heading, curr_heading), 1))
        dyaw = np.arccos(cosine)
        norm_v = np.cross(prev_heading, curr_heading)
        self.yaw += np.sign(norm_v)*dyaw
        
        if self.yaw > np.pi:
            self.yaw -= 2*np.pi
        if self.yaw < -np.pi:
            self.yaw += 2*np.pi
        self.heading = curr_heading
        yaw_dot = max(-30, min(dyaw/0.005, 30))
        return self.yaw, yaw_dot

    def get_des_state(self, t):
        pos, vel, acc = self.input_traj(t)
        yaw, yaw_dot = self.get_yaw(vel[:2].flatten())
        des_state = {'x': pos, 'v': vel,
                     'x_ddt': acc, 'yaw': yaw, 'yaw_dot': yaw_dot}
        return des_state
# trajectory=Trajectory('diamond')

# t=np.arange(0,17,0.01)
# x=[]
# v=[]
# a=[]
# for i in t:
#     des=trajectory.get_des_state(i)
#     x.append(des['x'])
#     v.append(des['v'])
#     a.append(des['x_ddt'])
# x=np.array(x)
# v=np.array(v)
# a=np.array(a)
# i = 1
# plt.subplot(3, 1, 1)
# plt.plot(t, x[:, i])
# plt.xlabel('t')
# plt.ylabel('state value')
# plt.title('x')
# plt.subplot(3, 1, 2)
# plt.plot(t, v[:, i])
# plt.xlabel('t')
# plt.ylabel('state value')
# plt.title('v')
# plt.subplot(3, 1, 3)
# plt.plot(t, a[:, i])
# plt.xlabel('t')
# plt.ylabel('state value')
# plt.title('acc')
# plt.show()
