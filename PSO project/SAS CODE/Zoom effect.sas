
proc IML ; 
Bornesup = 500; 
Borneinf = -500; 
pas = 10 ;
nbpas = bornesup/pas;
/*w=1;*/
Zoomresult = j(bornesup,3);
do u=1 to bornesup by pas; 
simu=20;
H=j(simu,4,0);
do y=1 to simu; 
c1 = 0;            /* Sous contrainte que (x1+x2 = 10)*/
c2 = 0.575; 
wmax =0.411; 
Nbvar =2;
wmin = 0.3; 
maxite=12;
nbpart=25;
maxx = Bornesup - u;
minx = Borneinf + u; 
*:x1max = Bornesup; 
/*x2max = Bor;*/
/*x1min = -500;*/
/*x2min = -500; */
g = j(nbpart,nbvar);  
s = j(nbpart,nbvar); 
call randgen(g, "Uniform"); 
call randgen(s,"Uniform"); 
/*Minx = min(x1min,x2min);
MAxx = max(x1max,x2max);*/   
x = Minx + (maxx-minx)*g; 
v= -25+ 50*s;
P=j(1,nbpart,0);  
do b=1 to nbpart;  
x1 = x[b,1];  
x2 = x[b,2];
do while(abs(x1)>maxx & abs(x2)>maxx);
o = j(nbpart,2);
call randgen(o,"uniform");
x[b,1] = Minx + (maxx-minx)*o[b,1];
x[b,2] = Minx + (maxx-minx)*o[b,2]; 
x1 = x[b,1];
x2 = x[b,2];
F = 418.9829*2 - x1*sin(abs(x1)**0.5) - x2*sin(abs(x2)**0.5); 
P[b] =F;
end;
F = 418.9829*2 - x1*sin(abs(x1)**0.5) - x2*sin((abs(x2)**0.5));
p[b] = F ; 
end; 
P = t(P);  
Gbest = Min(P);  
GPBEST = P[>:<];
Xbest = X; 
G1= X[GPBEST,1]; 
G2= X[GPBEST,2];
GM = G1||G2;
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
     X[c,1] = X[c,1] + v[c,1]; 
     X[c,2] = X[c,2] +v[c,2];
     x1 = x[c,1];
     x2 = x[c,2]; 
	 do while(abs(x1)>maxx & abs(x2)>maxx);
	 i = j(nbpart,2);
	 call randgen(i,"uniform");
     x[c,1] = Minx + (maxx-minx-1)*i[c,1];
     x[c,2] = minx + (maxx-minx-1)*i[c,2]; 
     x1 = x[c,1];
     x2 = x[c,2];
	 F = 418.9829*2 - x1*sin(abs(x1)**0.5) - x2*sin(abs(x2)**0.5); 
     P1[c] =F;
	 end;
     F = 418.9829*2 - x1*sin(abs(x1)**0.5) - x2*sin(abs(x2)**0.5); 
     P1[c] =F;
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
H[y,1]=Gbest;
H[y,2]=y;
H[y,3]=G1;
H[y,4]=G2;
P1 = t(P1);
end;
Result = H[1:simu,1];
Resultmin = Min(abs(H[1:simu,1]));
postimin = H[>:<,1];
G1min = H[postimin,3];
G2min = H[postimin,4];
Num = H[1:simu,2];
G1sim = H[1:simu,3];
G2sim = H[1:simu,4];
/*Zoomresult[w,1] = maxx; */
Zoomresult[u,1] = Resultmin; 
Zoomresult[u,2] = G1min; 
Zoomresult[u,3] = G2min;
end; 

print Zoomresult;  

/*MoyenG1 = sum(G1sim[1:simu,1])/simu;*/
/*MoyenG2 = sum(G2sim[1:simu,1])/simu;*/
/*resultmoy = sum(result[1:simu,1])/simu;*/
/*print H G1min G2min resultmin;*/
/*create pso.testcontrainte var{"result","num","G1sim","G2sim"};
append;
close pso.testcontrainte;*/
quit; 

