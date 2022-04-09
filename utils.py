from cProfile import label
import quadrotor
import controller
import trajectory
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
import csv

class Visualizer:
    def __init__(self, simu_time, simu_freq, ctrl_freq, real_trajectory, des_trajectory) -> None:
        self.simu_time = simu_time
        self.simu_freq = simu_freq
        self.ctrl_freq = ctrl_freq
        self.real_trajectory = real_trajectory
        self.des_trajectory = des_trajectory
        self.real_trajectory['x'] = np.array(self.real_trajectory['x'])
        self.real_trajectory['y'] = np.array(self.real_trajectory['y'])
        self.real_trajectory['z'] = np.array(self.real_trajectory['z'])
        self.des_trajectory['x'] = np.array(self.des_trajectory['x'])
        self.des_trajectory['y'] = np.array(self.des_trajectory['y'])
        self.des_trajectory['z'] = np.array(self.des_trajectory['z'])
        self.file_data = open('./data.csv', 'a')
        

    def animation_3d(self):
        fig = plt.figure()
        ax1 = p3.Axes3D(fig)  # 3D place for drawing
        point, = ax1.plot([self.real_trajectory['x'][0]], [self.real_trajectory['y'][0]], [self.real_trajectory['z'][0]], 'ro',
                        label='Quadrotor')
        line1, = ax1.plot([self.real_trajectory['x'][0]], [self.real_trajectory['y'][0]], [self.real_trajectory['z'][0]],
                        label='Real_Trajectory')
        line2, = ax1.plot(self.des_trajectory['x'][0], self.des_trajectory['y'][0], self.des_trajectory['z'][0],
                        label='Des_trajectory')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')
        ax1.set_xlim(-6,1)
        ax1.set_ylim(-3,3)
        ax1.set_zlim(-0,2.0)
        ax1.set_title('3D animate')
        ax1.view_init(35, 35)
        ax1.legend(loc='lower right')

        def animate(i):
            line1.set_xdata(self.real_trajectory['x'][:i + 10])
            line1.set_ydata(self.real_trajectory['y'][:i + 10])
            line1.set_3d_properties(self.real_trajectory['z'][:i + 10])
            line2.set_xdata(self.des_trajectory['x'][:i + 10])
            line2.set_ydata(self.des_trajectory['y'][:i + 10])
            line2.set_3d_properties(self.des_trajectory['z'][:i + 10])
            point.set_xdata(self.real_trajectory['x'][i])
            point.set_ydata(self.real_trajectory['y'][i])
            point.set_3d_properties(self.real_trajectory['z'][i])

        ani = animation.FuncAnimation(fig=fig,
                                    func=animate,
                                    frames=len(self.real_trajectory['x']),
                                    interval=1,
                                    repeat=False,
                                    blit=False)
        plt.show()
        
    def plot_obsv_x(self, x_real, x_obsv):
        t = np.arange(0, self.simu_time, 1/self.ctrl_freq)
        y1 = np.array(x_obsv[0])
        y2 = np.array(x_real[0])
        y3 = np.array(x_obsv[1])
        y4 = np.array(x_real[1])
        y5 = np.array(x_obsv[2])
        y6 = np.array(x_real[2])
        plt.plot(t, y1, '--', label="Vx_obsv")
        plt.plot(t, y2, label="Vx_real")
        plt.plot(t, y3, '--', label="Vy_obsv")
        plt.plot(t, y4, label="Vy_real")
        plt.plot(t, y5, '--', label="Vz_obsv")
        plt.plot(t, y6, label="Vz_real")
        plt.xlabel('time')
        plt.legend()
        plt.show()
    
    def plot_obsv_d(self, d_hat_list):
        t = np.arange(0, self.simu_time, 1/self.ctrl_freq)
        d_real = [0.02*9.81*0.2 for i in range(len(d_hat_list))]
        plt.plot(t, d_hat_list, label="d_hat")
        plt.plot(t, d_real, '--', label="d_real")
        plt.xlabel('time')
        plt.ylabel('disturbance')
        plt.legend()
        plt.show()
        
    def plot_tracking_performance(self):
        t = np.arange(0, self.simu_time, 1/self.simu_freq)
        plt.plot(t, self.real_trajectory['x'], label="x_real")
        plt.plot(t, self.real_trajectory['y'], label="y_real")
        plt.plot(t, self.real_trajectory['z'], label="z_real")
        plt.plot(t, self.des_trajectory['x'], '--', label="x_des")
        plt.plot(t, self.des_trajectory['y'], '--', label="y_des")
        plt.plot(t, self.des_trajectory['z'], '--', label="z_des")
        plt.title('real vs desired')
        plt.xlabel('time')
        plt.legend()
        plt.show()
    
    def record_tracking_data(self):
        writer = csv.writer(self.file_data)
        writer.writerow(self.real_trajectory['x'])
        writer.writerow(self.des_trajectory['x'])
        