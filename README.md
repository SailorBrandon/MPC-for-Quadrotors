# Quadrotor Trackers using MPC

Trajectory tracking controller using linear and nonlinear MPC.


## Dependencies
numpy, matplotlib, scipy, CVXPY, control, csv, acados

(Please follow the official [doc](https://docs.acados.org/python_interface/index.html) to install acados.)

## Usage
One can uncomment specific lines in  `runsim.py` to enable different controllers.
```
git clone https://github.com/SailorBrandon/MPC-for-Quadrotors.git
cd MPC-for-Quadrotors
python runsim.py
```