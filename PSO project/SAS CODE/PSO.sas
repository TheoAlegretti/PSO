
proc IML ; 
c1 = 0;
c2 = 0.575; 
wmax =0.411; 
Nbvar = 2;
wmin = 0.23; 
maxite=15;
nbpart=30;
g = j(nbpart,nbvar);  
s = j(nbpart,nbvar); 
call randgen(g, "Uniform"); 
call randgen(s,"Uniform"); 
Minx = -20;
MAxx = 20;   
x = Minx + (maxx-minx)*g; 
v= Minx +(maxx-minx)*s;
P=j(1,nbpart,0); 
do b=1 to nbpart;  
x1 = x[b,1];  
x2 = x[b,2];  
F= 2 + x1**2+ x2**2 ; 
P[b]= F;  
end; 
P = t(P); 
print P X V; 
Gbest = Min(P);  
GPBEST = P[>:<];
Xbest = X; 
G1= X[GPBEST,1]; 
G2= X[GPBEST,2];
GM = G1||G2;
print Gbest GPBEST GM; 
W = Wmax - ((wmax-wmin)/maxite); 
P1 = j(1,nbpart,0);  
do ite=1 to maxite;  
   do b=1 to nbpart; 
   Xbest[b,1] = Min(Xbest[b,1],X[b,1]); 
   Xbest[b,2] = Min(Xbest[b,2],X[b,2]);
   V[b,1] = W*v[b,1] + C1*uniform(0)*(Xbest[b,1]-X[b,1]) + C2*uniform(0)*(G1-X[b,1]); 
   V[b,2] = W*v[b,2] + C1*uniform(0)*(Xbest[b,2]-X[b,2]) + C2*uniform(0)*(G2-X[b,2]);
   end;
     do c=1 to nbpart;
     X[c,1] = X[c,1] +v[c,1]; 
     X[c,2] = X[c,2] +v[c,2];
     x1 = x[c,1];
     x2 = x[c,2]; 
     F = 2 + x1**2 + x2**2 ; 
     P1[c] = F; 
     end;
W = Wmax - ((wmax-wmin)/maxite)*ite; 
  if Min(P1)<=Gbest then do; 
  Gbest = Min(P1);
  GPbest = P1[>:<]; 
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
P1 = t(P1);
GM=G1||G2;
R = t(R);
print Gbest GPbest GM R R1 iteb; 
quit; 











