#PSO in free optimization # 

def PSO(maxite,minx,maxx,minvit,maxvit,nbsimulation):
    import random 
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    c1 = 0 
    c2 = 0.575
    wmax = 0.411
    wmin = 0.230
    parts = 30 
    #maxite = 300
    #minx = -220
    #maxx = 220
    #minvit = -5
    #maxvit = 5
    nbvar = 2 
    #nbsimulation = 20
    Simul_iteb = np.zeros(shape=(nbsimulation,1))
    Simul_gbest = np.zeros(shape=(nbsimulation,1))
    Simul_x1best = np.zeros(shape=(nbsimulation,1))
    Simul_x2best = np.zeros(shape=(nbsimulation,1))
    
    
    #Première itération d'initiation # 
    
    for simulation in range(0,nbsimulation,1):
        inx = np.zeros(shape=(parts,nbvar))
        inspeed = np.zeros(shape=(parts,nbvar))
        ValueF1 = np.zeros(shape=(parts,1))
        
        
        for nbv in range (0,nbvar,1):
            for r1 in range(0,parts,1): 
                  inx[r1,nbv] = random.uniform(minx, maxx) 
                  inspeed[r1,nbv] = random.randint(minvit,maxvit)
        for RVin in range(0,parts,1):
            ValueF1[RVin,0] = inx[RVin,0]**2 + inx[RVin,1]**2 + 2
            #fonction(inx[RVin,0]**2 + inx[RVin,1]**4 + 2)
        
        gbest = min(ValueF1)
        locc = np.argmin(ValueF1)
        W = wmax - ((wmax-wmin)/maxite)
        Xbest = inx
        ValueF = np.zeros(shape=(parts,1))
        
        iteb=0
        
        #on crée la boucle avec le nb d'itérations: 
        
        for ite in range(1,maxite,1):
            for r2 in range(0,parts,1):
                for nbv in range(0,nbvar,1): 
                    Xbest[r2,nbv] = min(Xbest[r2,nbv],inx[r2,nbv])
                    inspeed[r2,nbv] = W*inspeed[r2,nbv] + c1*random.random()*(Xbest[r2,nbv]-inx[r2,nbv]) + c2*random.random()*(inx[locc,nbv]-inx[r2,nbv])
                    inx[r2,nbv] = inx[r2,nbv] + inspeed[r2,nbv]
            for r3 in range(0,parts,1):
                ValueF[r3,0] = inx[r3,0]**2 + inx[r3,1]**2 + 2 
                #fonction(inx[r3,0]**2 + inx[r3,1]**4 + 2)
            W = wmax - ((wmax-wmin)/maxite)*ite
            if min(ValueF) <= gbest : 
                gbest = min(ValueF)
                locc = np.argmin(ValueF)
                iteb = ite
            else :
                gbest = gbest 
                locc = locc
        Simul_iteb[simulation,0] = iteb
        Simul_gbest[simulation,0] = gbest 
        Simul_x1best[simulation,0] = inx[locc,0]
        Simul_x2best[simulation,0] = inx[locc,1]
    
    moy_gbest = Simul_gbest.mean()
    min_gbest = Simul_gbest.min()
    loccsim = np.argmin(Simul_gbest)
    min_x1best = Simul_x1best[loccsim,0]
    min_x2best = Simul_x2best[loccsim,0]
    
    
   
    text0 = "We found in average :  "
    text1 = "Our solution is :  "
    text2 = "The x1* equals :  "
    text3 = "The x2* is :  "
    
    
    #Resultat_simulation.hist(1) #check ça l'aléatoire est chelou 
    print(text0)
    print(moy_gbest)
    print(text1) 
    print(min_gbest)
    print(text2)
    print(min_x1best)
    print(text3)
    print(min_x2best)
    
PSO(20,-5,5,-5,5,30)

#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################


#PSO machine learning with the zoom effect # 

#A Faire : l'intervalle de ML et contraintes 

def PSOML(maxite,minxZ,maxxZ,minvit,maxvit,nbsimulation):
    import random 
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    c1 = 0 
    c2 = 0.575
    wmax = 0.411
    wmin = 0.230
    parts = 30    
    nbvar = 2 
    Simul_iteb = np.zeros(shape=(nbsimulation,1))
    Simul_gbest = np.zeros(shape=(nbsimulation,1))
    Simul_x1best = np.zeros(shape=(nbsimulation,1))
    Simul_x2best = np.zeros(shape=(nbsimulation,1))
    
    
    #Première itération d'initiation # 
    #for 
    for simulation in range(0,nbsimulation,1):
        inx = np.zeros(shape=(parts,nbvar))
        inspeed = np.zeros(shape=(parts,nbvar))
        ValueF1 = np.zeros(shape=(parts,1))
        
        
        for nbv in range (0,nbvar,1):
            for r1 in range(0,parts,1): 
                  inx[r1,nbv] = random.uniform(minxZ, maxxZ) 
                  inspeed[r1,nbv] = random.randint(minvit,maxvit)
        for RVin in range(0,parts,1):
            ValueF1[RVin,0] = inx[RVin,0]**2 + inx[RVin,1]**2 + 2
            #fonction(inx[RVin,0]**2 + inx[RVin,1]**4 + 2)
        
        gbest = min(ValueF1)
        locc = np.argmin(ValueF1)
        W = wmax - ((wmax-wmin)/maxite)
        Xbest = inx
        ValueF = np.zeros(shape=(parts,1))
        
        iteb=0
        
        #on crée la boucle avec le nb d'itérations: 
        
        for ite in range(1,maxite,1):
            for r2 in range(0,parts,1):
                for nbv in range(0,nbvar,1): 
                    Xbest[r2,nbv] = min(Xbest[r2,nbv],inx[r2,nbv])
                    inspeed[r2,nbv] = W*inspeed[r2,nbv] + c1*random.random()*(Xbest[r2,nbv]-inx[r2,nbv]) + c2*random.random()*(inx[locc,nbv]-inx[r2,nbv])
                    inx[r2,nbv] = inx[r2,nbv] + inspeed[r2,nbv]
            for r3 in range(0,parts,1):
                ValueF[r3,0] = inx[r3,0]**2 + inx[r3,1]**2 + 2 
                #fonction(inx[r3,0]**2 + inx[r3,1]**4 + 2)
            W = wmax - ((wmax-wmin)/maxite)*ite
            if min(ValueF) <= gbest : 
                gbest = min(ValueF)
                locc = np.argmin(ValueF)
                iteb = ite
            else :
                gbest = gbest 
                locc = locc
        Simul_iteb[simulation,0] = iteb
        Simul_gbest[simulation,0] = gbest 
        Simul_x1best[simulation,0] = inx[locc,0]
        Simul_x2best[simulation,0] = inx[locc,1]
    
    moy_gbest = Simul_gbest.mean()
    min_gbest = Simul_gbest.min()
    loccsim = np.argmin(Simul_gbest)
    min_x1best = Simul_x1best[loccsim,0]
    min_x2best = Simul_x2best[loccsim,0]
    
    
   
    text0 = "We found in average :  "
    text1 = "Our solution is :  "
    text2 = "The x1* equals :  "
    text3 = "The x2* is :  "
    
    
    #Resultat_simulation.hist(1) #check ça l'aléatoire est chelou 
    print(text0)
    print(moy_gbest)
    print(text1) 
    print(min_gbest)
    print(text2)
    print(min_x1best)
    print(text3)
    print(min_x2best)
    

PSOML(5,-50,50,-5,5,30)


