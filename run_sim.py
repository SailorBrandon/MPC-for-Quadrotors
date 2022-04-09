from cProfile import label
import quadrotor
import controller
import trajectory
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
from utils import *
import time



# start simulation
if __name__=="__main__":
    quad_model = quadrotor.Quadrotor()
    quad_model.reset()
    real_trajectory = {'x': [], 'y': [], 'z': []}
    des_trajectory = {'x': [], 'y': [], 'z': []}
    
    accu_error_pos = np.zeros((3, ))
    total_time = 0
    square_ang_vel = np.zeros((4, ))
    
    simu_freq = 100 # Hz
    ctrl_freq = 50
    traj = trajectory.Trajectory("diamond")
    
    quad_controller = controller.Linear_MPC(traj, ctrl_freq, use_obsv=False)
    # quad_controller = controller.NonLinear_MPC(traj, ctrl_freq)
    # quad_controller = controller.PDcontroller(traj, ctrl_freq)

    simu_time = 10 # sec
    cur_time = 0
    dt = 1 / simu_freq
    num_iter = int(simu_time * simu_freq)
    start = time.time()
    for i in range(num_iter):
        des_state = traj.get_des_state(cur_time)
        if i % (int(simu_freq / ctrl_freq)) == 0:
            control_input, error_pos = quad_controller.control(cur_time, quad_model.state)
        cmd_rotor_speeds = control_input["cmd_motor_speeds"]
        obs, _, _, _ = quad_model.step(cmd_rotor_speeds)
        cur_time += dt

        accu_error_pos += error_pos * dt
        square_ang_vel += cmd_rotor_speeds ** 2 * dt
        
        real_trajectory['x'].append(obs['x'][0])
        real_trajectory['y'].append(obs['x'][1])
        real_trajectory['z'].append(obs['x'][2])
        
        des_trajectory['x'].append(des_state['x'][0][0])
        des_trajectory['y'].append(des_state['x'][1][0])
        des_trajectory['z'].append(des_state['x'][2][0])
    
    '''Print three required criterions'''
    print("Tracking performance: ", np.sum(accu_error_pos**2))
    print("Sum of square of angular velocities: ", np.sum(square_ang_vel))
    print("Total time: ", time.time() - start)
    
    '''Visualization'''
    visualizer = Visualizer(simu_time, simu_freq, ctrl_freq, real_trajectory, des_trajectory)
    visualizer.plot_tracking_performance()
    visualizer.record_tracking_data()
    try:
        if quad_controller.use_obsv == True:
            visualizer.plot_obsv_x(quad_controller.x_real, quad_controller.x_obsv)
            # visualizer.plot_obsv_d(quad_controller.d_hat_list)
    except:
        pass
    visualizer.animation_3d()
    
