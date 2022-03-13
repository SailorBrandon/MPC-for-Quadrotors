import quadrotor
import controller
import trajectory
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation

def visualization(real_trajectory, des_trajectory):
    fig = plt.figure()
    ax1 = p3.Axes3D(fig)  # 3D place for drawing
    real_trajectory['x'] = np.array(real_trajectory['x'])
    real_trajectory['y'] = np.array(real_trajectory['y'])
    real_trajectory['z'] = np.array(real_trajectory['z'])
    des_trajectory['x'] = np.array(des_trajectory['x'])
    des_trajectory['y'] = np.array(des_trajectory['y'])
    des_trajectory['z'] = np.array(des_trajectory['z'])
    point, = ax1.plot([real_trajectory['x'][0]], [real_trajectory['y'][0]], [real_trajectory['z'][0]], 'ro',
                    label='Quadrotor')
    line1, = ax1.plot([real_trajectory['x'][0]], [real_trajectory['y'][0]], [real_trajectory['z'][0]],
                    label='Real_Trajectory')
    line2, = ax1.plot(des_trajectory['x'][0], des_trajectory['y'][0], des_trajectory['z'][0],
                    label='Des_trajectory')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_xlim(-2,2)
    ax1.set_ylim(-2,2)
    ax1.set_zlim(-0,2.0)
    ax1.set_title('3D animate')
    ax1.view_init(35, 35)
    ax1.legend(loc='lower right')

    def animate(i):
        line1.set_xdata(real_trajectory['x'][:i + 5])
        line1.set_ydata(real_trajectory['y'][:i + 5])
        line1.set_3d_properties(real_trajectory['z'][:i + 5])
        line2.set_xdata(des_trajectory['x'][:i + 5])
        line2.set_ydata(des_trajectory['y'][:i + 5])
        line2.set_3d_properties(des_trajectory['z'][:i + 5])
        point.set_xdata(real_trajectory['x'][i])
        point.set_ydata(real_trajectory['y'][i])
        point.set_3d_properties(real_trajectory['z'][i])

    ani = animation.FuncAnimation(fig=fig,
                                func=animate,
                                frames=len(real_trajectory['x']),
                                interval=1,
                                repeat=False,
                                blit=False)
    plt.show()

# start simulation
if __name__=="__main__":
    quad_model = quadrotor.Quadrotor()
    quad_model.reset()
    simu_freq = 100 # Hz
    ctrl_freq = 50
    traj = trajectory.Trajectory("diamond")
    quad_controller = controller.Linear_MPC(traj, ctrl_freq)
    # quad_controller = controller.PDcontroller(traj, ctrl_freq)
    real_trajectory = {'x': [], 'y': [], 'z': []}
    des_trajectory = {'x': [], 'y': [], 'z': []}
    
    # initialize performance matrics
    # accu_error_pos = np.zeros((3, 1))
    # error_pos = np.zeros((3, 1))
    # total_time = 0
    # square_ang_vel = np.zeros((4, ))
    
    simu_time = 10 # sec
    cur_time = 0
    dt = 1 / simu_freq
    num_iter = int(simu_time * simu_freq)
    for i in range(num_iter):
        des_state = traj.get_des_state(cur_time)
        if i % (int(simu_freq / ctrl_freq)) == 0:
            control_input, error_pos = quad_controller.control(cur_time, quad_model.state)
        cmd_rotor_speeds = control_input["cmd_motor_speeds"]
        obs, _, _, _ = quad_model.step(cmd_rotor_speeds)
        cur_time += dt
        # # check whether the terminal point is reached
        # if np.all(abs(obs['x'] - final_pos.reshape(obs['x'].shape)) < np.full((3, 1), 1e-2)):
        #     total_time = cur_time
        #     break
        # else:
        #     total_time = simu_time
        # record performance matrics
        # accu_error_pos += error_pos * dt
        # square_ang_vel += cmd_rotor_speeds ** 2 * dt
        # record trajectories for visualization
        real_trajectory['x'].append(obs['x'][0])
        real_trajectory['y'].append(obs['x'][1])
        real_trajectory['z'].append(obs['x'][2])
        des_trajectory['x'].append(des_state['x'][0][0])
        des_trajectory['y'].append(des_state['x'][1][0])
        des_trajectory['z'].append(des_state['x'][2][0])
    
    # Print three required criterions
    # print("Tracking performance: ", np.sum(accu_error_pos**2))
    # print("Total time needed: ", total_time)
    # print("Sum of square of angular velocities: ", np.sum(square_ang_vel))

    # Visualization
    start_ani = 1
    if (start_ani):
        visualization(real_trajectory, des_trajectory)
