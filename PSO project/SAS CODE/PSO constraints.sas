proc IML ; 
simu=200; /* Number of monte carlo simulations */ 
H=j(simu,4,0);
do y=1 to simu; 
/* Optimal config */ 
c1 = 0;           
c2 = 0.575; 
wmax =0.411; 
Nbvar =2; /*Number of variables on the objective function */ 
wmin = 0.23; 
maxite=15;
nbpart=20;
x1max = 15; /* x1 max for the saturation of the constraint (when x2 = 0) */ 
x2max = 7.5; /* x2 max for the saturation of the constraint (when x1 = 0) */
x1min = 0; 
X2min = 0; 
g = j(nbpart,nbvar);  
s = j(nbpart,nbvar); 
call randgen(g, "Uniform"); 
call randgen(s,"Uniform"); 
Minx = min(x1min,x2min);
MAxx = max(x1max,x2max);   
x = Minx + (maxx-minx)*g; 
v= Minx +(maxx-minx)*s;
P=j(1,nbpart,0);  
do b=1 to nbpart;  
x1 = x[b,1];  
x2 = x[b,2];
do while(x1*100+x2*200>=1500); /* The constraint with no negativity */
x[b,1] = maxx * uniform(0);
x[b,2] = maxx * uniform(0); 
x1 = x[b,1];
x2 = x[b,2];
F = (x1**2)*(x2**3); /* Objective function */
P[b] =F;
end;
F = (x1**2) *(x2 **3); /*Objective function */ 
p[b] = F ; 
end; 
P = t(P);  
Gbest = Max(P);  
GPBEST = P[<:>];
Xbest = X; 
G1= X[GPBEST,1]; 
G2= X[GPBEST,2];
GM = G1||G2;
W = Wmax - ((wmax-wmin)/maxite); 
P1 = j(1,nbpart,0);  
do ite=1 to maxite;  
   do b=1 to nbpart; 
   Xbest[b,1] = Max(Xbest[b,1],X[b,1]); 
   Xbest[b,2] = Max(Xbest[b,2],X[b,2]);
   V[b,1] = W*v[b,1] + C1*uniform(0)*(Xbest[b,1]-X[b,1]) + C2*uniform(0)*(G1-X[b,1]); 
   V[b,2] = W*v[b,2] + C1*uniform(0)*(Xbest[b,2]-X[b,2]) + C2*uniform(0)*(G2-X[b,2]);
   end;
     do c=1 to nbpart;
     X[c,1] = X[c,1] + v[c,1]; 
     X[c,2] = X[c,2] +v[c,2];
     x1 = x[c,1];
     x2 = x[c,2]; 
	 do while(x1*100+x2*200>=1500); /* The constraint with no negativity */
	 x[c,1] = maxx * uniform(0) ;
     x[c,2] = maxx * uniform(0) ;
	 x1 = x[c,1];
	 x2 = x[c,2];
     F = (x1**2)*(x2**3); /*Objective function */ 
     P1[c] = F; 
     end;
	 F = (x1**2) *(x2**3); /*Objective function */ 
	 P1[c] = F;   
	 end;	 
W = Wmax - ((wmax-wmin)/maxite)*ite; 
  if Max(P1)>=Gbest then do; 
  Gbest = Max(P1);
  GPbest = P1[<:>]; 
  G1= X[GPbest,1];
  G2= X[GPbest,2];
  R=P1; 
  R1 = X; 
  iteb = ite; 
  end;
    else do;
    Gbest = Gbest; 
    G1=G1;
    G2=G2; 
    end; 
  end;
  H[y,1]=Gbest;
H[y,2]=y;
H[y,3]=G1;
H[y,4]=G2;
P1 = t(P1);
end;
Result = H[1:simu,1];
Num = H[1:simu,2];
G1sim = H[1:simu,3];
G2sim = H[1:simu,4];
MoyenG1 = sum(G1sim[1:simu,1])/simu;
MoyenG2 = sum(G2sim[1:simu,1])/simu;
resultmoy = sum(result[1:simu,1])/simu;
print H result resultmoy moyenG1 moyenG2;
/*create pso.testcontrainte var{"result","num","G1sim","G2sim"};
append;
close pso.testcontrainte;*/
call histogram(Result);
quit; 
