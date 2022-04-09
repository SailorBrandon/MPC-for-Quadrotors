function[xr,ur]=optimalss(LTI,dim,weight,constraints,eqconstraints)

H=blkdiag(zeros(dim.nx),eye(dim.nu));
h=zeros(dim.nx+dim.nu,1);


options1 = optimoptions(@quadprog); 
options1.OptimalityTolerance=1e-20;
options1.ConstraintTolerance=1.0000e-15;
options1.Display='off';
xur=quadprog(H,h,[],[],eqconstraints.A,eqconstraints.b,[],[],[],options1);
xr=xur(1:dim.nx);
ur=xur(dim.nx+1:end);

end