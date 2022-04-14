import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


class Terminal_set:
    def __init__(self, Hx, Hu, K, Ak, h):
        self.dt = 0.01
        self.Hx = Hx
        self.Hu = Hu
        self.K = K
        self.Ak = Ak
        self.H = np.block([[Hu, np.zeros((Hu.shape[0], Hx.shape[1]))],
                          [np.zeros((Hx.shape[0], Hu.shape[1])), Hx]])
        self.Nc = self.H.shape[0]
        self.Nx = Ak.shape[1]
        self.h = h
        self.K_aug = np.vstack((-K, np.eye(self.Nx)))
        self.maxiter = 200
        self.Xf = self.terminal_set_cal()
        # self.Xf_nr = self.Xf
        self.Xf_nr = self.remove_redundancy()
        # self.LQR_control()
        self.test_input_inbound(0.15)

    def terminal_set_cal(self):
        Ainf = np.zeros([0, self.Nx])
        binf = np.zeros([0, 1])
        Ft = np.eye(self.Nx)
        # gravity=np.zeros([12, 1])
        # gravity[6]=self.dt*9.801
        # gravity[11]=self.dt*9.801
        self.C = self.H@self.K_aug

        for t in range(self.maxiter):
            Ainf = np.vstack((Ainf, self.C@Ft))
            # binf = np.vstack((binf, self.h-gravity*(t+1)))
            binf = np.vstack((binf, self.h))
            # print(self.h-gravity*(t+1))
            Ft = self.Ak@Ft
            fobj = self.C@Ft
            violation = False
            for i in range(self.Nc):
                val, x = self.solve_linprog(fobj[i, :], Ainf, binf)
                if val > self.h[i]:
                    violation = True
                    break
            if not violation:
                return [Ainf, binf]

    def solve_linprog(self, obj, Ainf, binf):
        x = cp.Variable((self.Nx, 1))
        objective = cp.Maximize(obj@x)
        constraints = [Ainf@x <= binf]
        linear_program = cp.Problem(objective, constraints)

        result = linear_program.solve(verbose=False)

        # print('result=',result)
        # print('x=',x.value)
        return result, x.value

    def remove_redundancy(self):
        A_inf, Binf = self.Xf
        Ainf_nr, binf_nr = A_inf.copy(), Binf.copy()
        i = 0
        while i < Ainf_nr.shape[0]:
            obj = Ainf_nr[i, :]
            binf_temp = binf_nr.copy()
            binf_temp[i] += 1
            val, x = self.solve_linprog(obj, Ainf_nr, binf_temp)
            if val < binf_nr[i] or val == binf_nr[i]:
                Ainf_nr = np.delete(Ainf_nr, i, 0)
                binf_nr = np.delete(binf_nr, i, 0)
            else:
                i += 1
        return [Ainf_nr, binf_nr]

    def LQR_control(self, N=100):
        A_inf, b_inf = self.Xf_nr
        vertices = []
        for i in range(A_inf.shape[0]):
            obj = A_inf[i, :]
            _, x = self.solve_linprog(obj, A_inf, b_inf)
            vertices.append(x)
        vertices = np.array(vertices)
        states = np.zeros([N, vertices.shape[0], 12])
        inputs = np.zeros([N, vertices.shape[0], 4])
        for index in range(vertices.shape[0]):
            x = vertices[index].copy()
            for t in range(N):
                states[t, index, :] = x.squeeze()
                u = -self.K@x
                inputs[t, index, :] = u.squeeze()
                x = self.Ak@x
                x[5] -= 9.81*0.01
        t = np.arange(N)
        states = states.transpose((2, 1, 0))
        inputs = inputs.transpose((2, 1, 0))
        i = 0
        vertex = 42
        plt.subplot(2, 3, 1)
        plt.plot(t, states[0+i, vertex, :])
        plt.xlabel('t')
        plt.ylabel('state value')
        plt.title('x')
        plt.subplot(2, 3, 2)
        plt.plot(t, states[i+1, vertex, :])
        plt.xlabel('t')
        plt.ylabel('state value')
        plt.title('y')
        plt.subplot(2, 3, 3)
        plt.plot(t, states[i+2, vertex, :])
        plt.xlabel('t')
        plt.ylabel('state value')
        plt.title('z')
        plt.subplot(2, 3, 4)
        plt.plot(t, states[i+3, vertex, :])
        plt.xlabel('t')
        plt.ylabel('state value')
        plt.title('vx')
        plt.subplot(2, 3, 5)
        plt.plot(t, states[i+4, vertex, :])
        plt.xlabel('t')
        plt.ylabel('state value')
        plt.title('vy')
        plt.subplot(2, 3, 6)
        plt.plot(t, states[i+5, vertex, :])
        plt.xlabel('t')
        plt.ylabel('state value')
        plt.title('vz')
        plt.show()
        input()
        
    def test_input_inbound(self,u_limit):
        A_inf,b_inf=self.Xf_nr
        violation =False
        for i in range(4):
            x = cp.Variable((12, 1))
            u = cp.Variable((4, 1))
            cost=0
            constr = []
            constr.append(A_inf@x[:,0] <= b_inf.squeeze())
            constr.append(u[:, 0]==-self.K@x[:,0])
            # constr.append(self.Ak@x[:,0]==x[:,1])
            cost=u[i, 0]
            # cost+= cp.quad_form(self.Ak@x[:,0], P)
            # cost-=cp.quad_form(x[:, 0], P)
            # cost+=(cp.quad_form(x[:, 0], Q) + cp.quad_form(u[:, 0], R))
            problem = cp.Problem(cp.Maximize(cost), constr)
            problem.solve()
            print('Input u',i,'<',problem.value)
            if problem.value >u_limit:
                violation =True
        if violation ==False:
            print('Input inbound')
# Hu=np.array([[1,0],[-1,0],[0,1],[0,-1]])
# Hx=np.array([[1,0],[-1,0]])
# K=np.array([[-1.35,-0.9],[-0.225,-1.65]])
# Ak=np.array([[0.65,0.1],[-0.225,0.35]])
# h=np.array([[1],[1],[1],[1],[5],[5]])
# Ter_set=Terminal_set(Hx, Hu, K, Ak,h)
# print('A_inf',Ter_set.Xf[0])
# print('b_inf',Ter_set.Xf[1])
# print('Ainf_nr',Ter_set.Xf_nr[0])
# print('binf_nr',Ter_set.Xf_nr[1])

    # def test_lya_decrease(self, P, Q, R, N=25):
    #     A_inf, b_inf = self.Xf_nr
    #     vertices = []
    #     for i in range(A_inf.shape[0]):
    #         obj = A_inf[i, :]
    #         _, x = self.solve_linprog(obj, A_inf, b_inf)
    #         vertices.append(x)
    #     vertices = np.array(vertices)
