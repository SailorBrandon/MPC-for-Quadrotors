import numpy as np
import numpy.linalg as LA


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

    def diamond(self, t):
        T1, T2, T3, T4 = 3, 3, 3, 3
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
        T1 = 0.8
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
