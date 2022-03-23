import numpy as np
import cvxpy as cp
class Terminal_set:
    def __init__(self, Hx, Hu, K, Ak,h):
        self.Hx = Hx
        self.Hu = Hu
        self.K = K
        self.Ak = Ak
        self.H=np.block([[Hu, np.zeros((Hu.shape[0], Hx.shape[1]))],
                        [np.zeros((Hx.shape[0], Hu.shape[1])), Hx]])
        self.Nc= self.H.shape[0]
        self.Nx= Ak.shape[1]
        self.h=h
        self.K_aug=np.vstack((K,np.eye(self.Nx)))
        self.maxiter=30
        self.Xf=self.terminal_set_cal()
        self.Xf_nr=self.remove_redundancy()
    def terminal_set_cal(self):
        Ainf=np.zeros([0,self.Nx])
        binf=np.zeros([0,1])
        Ft=np.eye(self.Nx)
        self.C=self.H@self.K_aug

        for t in range(self.maxiter):
            Ainf = np.vstack((Ainf,self.C@Ft))
            binf=np.vstack((binf,self.h))
            Ft=self.Ak@Ft
            fobj=self.C@Ft
            violation=False
            for i in range(self.Nc):
                val,x=self.solve_linprog(fobj[i,:],Ainf,binf)
                if val>self.h[i]:
                    violation = True
                    break
            if not violation:
                return [Ainf,binf]

    def solve_linprog(self,obj,Ainf,binf):
        x = cp.Variable((self.Nx,1))
        objective = cp.Maximize(obj@x)
        constraints = [Ainf@x <= binf]
        linear_program = cp.Problem(objective, constraints)
        result = linear_program.solve()
        # print('result=',result)
        # print('x=',x.value)
        return result,x.value
    
    def remove_redundancy(self):
        A_inf,Binf=self.Xf
        Ainf_nr,binf_nr=A_inf.copy(),Binf.copy()
        i=0
        while i< Ainf_nr.shape[0]:
            obj=Ainf_nr[i,:]
            binf_temp=binf_nr.copy()
            binf_temp[i]+=1
            val,x=self.solve_linprog(obj,Ainf_nr,binf_temp)
            if val<binf_nr[i] or val==binf_nr[i]:
                Ainf_nr=np.delete(Ainf_nr,i,0)
                binf_nr=np.delete(binf_nr,i,0)
            else:
                i+=1
        return [Ainf_nr,binf_nr]

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