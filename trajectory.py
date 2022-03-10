import numpy as np

# start_pos 3 by 1 vector
# pos 3 by 1 vector

# Hover Jrajectory
def hover(t):
    pos, pos_dot, pos_ddt, final_pos = np.zeros((3, 1)), np.zeros((3, 1)), np.zeros((3, 1)), np.zeros((3, 1))
    return pos, pos_dot, pos_ddt, final_pos


# Connecting waypoints through Bang-Bang trajectory
def tj_from_line(start_pos, end_pos, time_ttl, t_c):
    v_max = (end_pos - start_pos) * 2 / time_ttl
    if (t_c >= 0 and t_c < time_ttl/2):
        pos_dot = v_max*t_c/(time_ttl/2)
        pos = start_pos + t_c*pos_dot/2
        pos_ddt = v_max/(time_ttl/2)
    elif (t_c >= time_ttl/2 and t_c <= time_ttl):
        pos_dot = v_max * (time_ttl - t_c) / (time_ttl / 2)
        pos = end_pos - (time_ttl - t_c) * pos_dot / 2
        pos_ddt = - v_max/(time_ttl / 2)
    else:
        if (type(start_pos) == int) or (type(start_pos) == float):
            pos, pos_dot, pos_ddt = 0, 0, 0
        else:
            pos, pos_dot, pos_ddt = np.zeros(start_pos.shape), np.zeros(
                start_pos.shape), np.zeros(start_pos.shape)
    return pos, pos_dot, pos_ddt


# Diamond Jrajectory
# Quadrotor navigates through 4 given waypoints
def diamond(t):
    T1, T2, T3, T4 = 3, 3, 3, 3
    points = []
    points.append(np.zeros((3, 1)))
    points.append(np.array([[0], [np.sqrt(2)], [np.sqrt(2)]]))
    points.append(np.array([[1], [0], [2*np.sqrt(2)]]))
    points.append(np.array([[1], [-np.sqrt(2)], [np.sqrt(2)]]))
    points.append(np.array([[1], [0], [0]]))
    if (0 < t) and (t <= T1):
        pos, pos_dot, pos_ddt = tj_from_line(points[0], points[1], T1, t)
    elif (T1 < t) and (t <= (T1+T2)):
        pos, pos_dot, pos_ddt = tj_from_line(points[1], points[2], T2, t-T1)
    elif ((T1 + T2) < t) and (t <= (T1 + T2 + T3)):
        pos, pos_dot, pos_ddt = tj_from_line(points[2], points[3], T3, t - (T1 + T2))
    elif ((T1 + T2 + T3) < t) and (t <= (T1 + T2 + T3 + T4)):
        pos, pos_dot, pos_ddt = tj_from_line(
            points[3], points[4], T4, t - (T1 + T2 + T3))
    elif (t > (T1 + T2 + T3 + T4)):
        pos, pos_dot, pos_ddt = points[4], np.zeros((3, 1)), np.zeros((3, 1))
    else:
        pos, pos_dot, pos_ddt = np.zeros((3, 1)), np.zeros((3, 1)), np.zeros((3, 1))
    final_pos = points[-1]
    return pos, pos_dot, pos_ddt, final_pos

def ascent(t):
    T1 = 3
    points = []
    points.append(np.zeros((3, 1)))
    points.append(np.array([[0], [0], [3]]))
    if (0 < t) and (t <= T1):
        pos, pos_dot, pos_ddt = tj_from_line(points[0], points[1], T1, t)
    else:
        pos, pos_dot, pos_ddt = points[-1], np.zeros((3, 1)), np.zeros((3, 1))
    final_pos = points[-1]
    return pos, pos_dot, pos_ddt, final_pos

# cmd_state
# x, x_dot, x_ddot, yaw, yaw_dot
def generate_trajec(input_traj, t=0):
    pos, pos_dot, pos_ddt, final_pos = input_traj(t)
    yaw = 0
    yaw_dot = 0
    cmd_state = {'x': pos, 'x_dot': pos_dot,
                 'x_ddt': pos_ddt, 'yaw': yaw, 'yaw_dot': yaw_dot}
    return cmd_state, final_pos
