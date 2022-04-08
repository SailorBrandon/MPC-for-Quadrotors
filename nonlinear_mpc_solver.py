import numpy as np
from dataclasses import dataclass
import casadi as cd
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from quadrotor import Quadrotor



@dataclass
class MPC_Formulation_Param:
    quad = Quadrotor()
    mass = quad.mass
    g = quad.g

    # dynamics
    Ixx = quad.Ixx
    Iyy = quad.Iyy
    Izz = quad.Izz

    # control bound
    thrust_min =  0.5 * g            # mass divided
    thrust_max = 1.5 * g
    roll_max = np.deg2rad(25)
    pitch_max = np.deg2rad(25)
    yaw_max = np.deg2rad(60)

    
    # state cost weights
    q_x = 80
    q_y = 80
    q_z = 120
    q_vx = 80
    q_vy = 80
    q_vz = 100
    q_roll = 50
    q_pitch = 50
    q_yaw = 50
    q_roll_rate = 10
    q_pitch_rate = 10
    q_yaw_rate = 10
    r_thrust = 1
    r_roll = 50
    r_pitch = 50
    r_yaw = 50
    
    # terminal cost weights
    q_x_terminal = 80
    q_y_terminal = 80
    q_z_terminal = 120
    q_vx_terminal = 80
    q_vy_terminal = 80
    q_vz_terminal = 100
    
    def set_horizon(self, dt, N):
        self.dt = dt
        self.N = N
        self.Tf = N * dt 
    
    


def acados_mpc_solver_generation(mpc_form_param, collision_avoidance = False):
    # Acados model
    model = AcadosModel()
    model.name = "mav_nmpc_tracker_model"

    # state
    px = cd.MX.sym('px')
    py = cd.MX.sym('py')
    pz = cd.MX.sym('pz')
    vx = cd.MX.sym('vx')
    vy = cd.MX.sym('vy')
    vz = cd.MX.sym('vz')
    roll = cd.MX.sym('roll')
    pitch = cd.MX.sym('pitch')
    yaw = cd.MX.sym('yaw')
    roll_rate = cd.MX.sym('roll_rate')
    pitch_rate = cd.MX.sym('pitch_rate')
    yaw_rate = cd.MX.sym('yaw_rate')
    x = cd.vertcat(px, py, pz, vx, vy, vz, roll, pitch,
                   yaw, roll_rate, pitch_rate, yaw_rate)

    # control
    thrust_cmd = cd.MX.sym('thrust_cmd')  # thrust_cmd = thrust/mass
    roll_cmd = cd.MX.sym('roll_cmd')
    pitch_cmd = cd.MX.sym('pitch_cmd')
    yaw_cmd = cd.MX.sym('yaw_cmd')
    u = cd.vertcat(thrust_cmd, roll_cmd, pitch_cmd, yaw_cmd)

    # state derivative
    px_dot = cd.MX.sym('px_dot')
    py_dot = cd.MX.sym('py_dot')
    pz_dot = cd.MX.sym('pz_dot')
    vx_dot = cd.MX.sym('vx_dot')
    vy_dot = cd.MX.sym('vy_dot')
    vz_dot = cd.MX.sym('vz_dot')
    roll_dot = cd.MX.sym('roll_dot')
    pitch_dot = cd.MX.sym('pitch_dot')
    yaw_dot = cd.MX.sym('yaw_dot')
    roll_rate_dot = cd.MX.sym('roll_rate_dot')
    pitch_rate_dot = cd.MX.sym('pitch_rate_dot')
    yaw_rate_dot = cd.MX.sym('yaw_rate_dot')
    x_dot = cd.vertcat(px_dot, py_dot, pz_dot, vx_dot, vy_dot, vz_dot, roll_dot,
                       pitch_dot, yaw_dot, roll_rate_dot, pitch_rate_dot, yaw_rate_dot)

    # dynamics
    dyn_f_expl = cd.vertcat(
        vx,
        vy,
        vz,
        (cd.cos(yaw) * cd.sin(pitch) + cd.cos(pitch) * cd.sin(roll) * cd.sin(yaw)) * thrust_cmd,
        (cd.sin(yaw) * cd.sin(pitch) - cd.cos(yaw) * cd.cos(pitch) * cd.sin(roll)) * thrust_cmd,
        -mpc_form_param.g + cd.cos(pitch) * cd.cos(roll) * thrust_cmd,
        roll_rate,
        pitch_rate,
        yaw_rate,
        (mpc_form_param.Izz - mpc_form_param.Iyy) * pitch_rate * yaw_rate/ mpc_form_param.Ixx + roll_cmd / mpc_form_param.Ixx,
        (mpc_form_param.Ixx - mpc_form_param.Izz) * roll_rate * yaw_rate/ mpc_form_param.Iyy + pitch_cmd / mpc_form_param.Iyy,
        (mpc_form_param.Iyy - mpc_form_param.Ixx) * roll_rate * pitch_rate/ mpc_form_param.Izz + yaw_cmd / mpc_form_param.Izz,
    )
    dyn_f_impl = x_dot - dyn_f_expl

    
    # acados mpc model
    model.x = x
    model.u = u
    model.xdot = x_dot
    model.f_expl_expr = dyn_f_expl
    model.f_impl_expr = dyn_f_impl

    # Acados ocp
    ocp = AcadosOcp()
    ocp.model = model

    # ocp dimension
    ocp.dims.N = mpc_form_param.N
    nx = 12
    nu = 4
    ny = nx + nu  # TODO
    ny_e = 6  # terminal cost, penalize on pos, pos_dot

    # initial condition, can be changed in real time
    ocp.constraints.x0 = np.zeros(nx)

    # cost terms
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    Vx = np.block([[np.eye(nx)],
                   [np.zeros((nu, nx))]])
    ocp.cost.Vx = Vx
    Vu = np.block([[np.zeros((nx, nu))],
                   [np.eye(nu)]]) 
    ocp.cost.Vu = Vu
    Vx_e = np.block([np.eye(6), np.zeros((6, 6))]) # Hard coded for now
    ocp.cost.Vx_e = Vx_e
    # weights, changed in real time
    ocp.cost.W = np.diag([mpc_form_param.q_x, mpc_form_param.q_y, mpc_form_param.q_z,
                          mpc_form_param.q_vx, mpc_form_param.q_vy, mpc_form_param.q_vz,
                          mpc_form_param.q_roll, mpc_form_param.q_pitch, mpc_form_param.q_yaw,
                          mpc_form_param.q_roll_rate, mpc_form_param.q_pitch_rate, mpc_form_param.q_yaw_rate,
                          mpc_form_param.r_thrust, mpc_form_param.r_roll, mpc_form_param.r_pitch, mpc_form_param.r_yaw])
    ocp.cost.W_e =  np.diag([mpc_form_param.q_x_terminal, mpc_form_param.q_y_terminal, mpc_form_param.q_z_terminal,
                          mpc_form_param.q_vx_terminal, mpc_form_param.q_vy_terminal, mpc_form_param.q_vz_terminal])

    # collision avoidance
    obstacle = np.array([0, 0.5, 0.5])
    distance = 0.3
    con_h_expr = (px - obstacle[0])**2 + (py - obstacle[1])**2 + (pz - obstacle[2])**2
    if collision_avoidance == True:
        model.con_h_expr = con_h_expr
        ocp.constraints.lh = np.array([distance**2])
        ocp.constraints.uh = np.array([10 * distance**2])
        ocp.cost.Zl = 100000 * np.array([1])
        ocp.constraints.Jsh = np.array([[1]])
        ocp.cost.Zu = np.array([0])
        ocp.cost.zu = np.array([0])
        ocp.cost.zl = np.array([0])
    
    
    # reference for tracking, changed in real time
    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    # set control bound
    ocp.constraints.lbu = np.array(
        [mpc_form_param.thrust_min, -mpc_form_param.roll_max, -mpc_form_param.pitch_max, -mpc_form_param.yaw_max])
    ocp.constraints.ubu = np.array(
        [mpc_form_param.thrust_max, mpc_form_param.roll_max, mpc_form_param.pitch_max, mpc_form_param.yaw_max])
    ocp.constraints.idxbu = np.array(range(nu))

    # solver options
    # horizon
    ocp.solver_options.tf = mpc_form_param.Tf
    # qp solver
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'    # PARTIAL_CONDENSING_HPIPM FULL_CONDENSING_QPOASES
    ocp.solver_options.qp_solver_cond_N = 5
    ocp.solver_options.qp_solver_iter_max = 100
    ocp.solver_options.qp_solver_warm_start = 1
    # nlp solver
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.nlp_solver_max_iter = 100
    ocp.solver_options.nlp_solver_tol_eq = 1E-3
    ocp.solver_options.nlp_solver_tol_ineq = 1E-3
    ocp.solver_options.nlp_solver_tol_comp = 1E-3
    ocp.solver_options.nlp_solver_tol_stat = 1E-3
    # hessian
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    # integrator
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 3
    # print
    ocp.solver_options.print_level = 0

    # Acados solver
    print("Starting solver generation...")
    solver = AcadosOcpSolver(ocp, json_file='ACADOS_nmpc_tracker_solver.json')
    print("Solver generated.")

    return solver


if __name__ == "__main__":
    param = MPC_Formulation_Param()
    acados_mpc_solver_generation(param)
