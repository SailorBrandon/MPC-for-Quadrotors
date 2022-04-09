function [H,h]=costgen(predmod,weight,dim)

Qbar=blkdiag(kron(eye(dim.N),weight.Q),weight.P);
Rbar=kron(eye(dim.N),weight.R);
H=predmod.S'*Qbar*predmod.S+Rbar;   
hx0=predmod.S'*Qbar*predmod.T;
hxref=-predmod.S'*Qbar*kron(ones(dim.N+1,1),eye(dim.nx));
huref=-Rbar*kron(ones(dim.N,1),eye(dim.nu));
h=[hx0 hxref huref];
 
end