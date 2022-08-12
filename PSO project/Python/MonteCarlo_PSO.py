
import random 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import math as mt 
import scipy.stats as sp
import plotly.figure_factory as ff


df = pd.read_csv(r'/Users/theoalegretti/Desktop/_VIZ_PRICE_ELASTICITY_PREDICT__202208091108.csv',sep=',')

def graph(data): 
    fig = make_subplots(rows=1, cols=2,subplot_titles=(f"Histogram of OCC rate error ", f"Histogram of CC ouv error","Test de "))
    fig.add_trace(go.Histogram(x=data['ERROR_OCC'],histnorm='probability density', name='Error_occ'),row=1,col=1)
    fig.add_trace(go.Histogram(x=data['ERROR_CC'],histnorm='probability density', name='Error_cc'),row=1,col=2)
    print(len(data))
    fig.show()
    
graph(df.sample(frac = 1))

graph(df[df['ID_OD']==642])

x = df['ERROR_CC']
mean = x.mean()
std = x.std()
x=np.linspace(-3,3,200)

y = sp.norm.cdf(x,mean,std)

fig = px.histogram(x=y)
fig.show()


def gaussian_density(x,mean,std):
    return 1/(std*(np.sqrt(2*np.pi)))*np.exp(-0.5*(((x-mean)/std)**2))


# test avec les paramètres de stat descriptive
Simu = df['ERROR_OCC'].dropna()
mean = Simu.mean()
std = Simu.std()
plt.hist(Simu,bins=300,density=True,edgecolor="k")
x=np.linspace(min(Simu),max(Simu),2000)
plt.plot(x, gaussian_density(x,mean,std))
plt.show()



x=np.linspace(min(Simu),max(Simu),2000,endpoint=True)

y = []
y.append(np.log(sp.norm.pdf(Simu,x1,x2)).sum())

plt.plot(x,y)
plt.title(r'Log-Likelihood')
plt.xlabel(r'$\mu$')
plt.grid()
plt.show()






def pso(parts,vit,c1,c2,Simu) : 
    global Xite,Yite,gbest,inspeed,nbvar,maxx,minx,Vite,locc,allgb,fctname,allgbp,inx,ValueF,df,maxite
    #Les paramètres que l'on fait bouger avec l'application => PARAMETRISATION 
    #c1 = 0 #liberté 
    #c2 = 0.575 #dépendance
    #wmax = 0.411 optimisé => Permet le calcul de la vitesse 
    #wmin = 0.23 optimisé  => Permet le calcul de la vitesse 
    #parts = 30 
    #wmax = 0.411
    #wmin = 0.23
    #Ici j'ai appelé qu'un paramètre vit qui me permet de construire la vitesse avec une décomposition simple (+ pratique)
    wmax = vit + vit/2
    wmin = vit - vit/2
    #J'ai fixé le nombre d'itération à 50 arbitrairement mais on peut l'augmenter ou le descendre (c'est le nombre de déplacement maximum des oiseaux)
    maxite = 50 #peut être modifier 
    #Je vais fixé les ensembles de définitions de chaqu'une des fonctions, cela va permettre de fixé les graphiques dans un premier temps 
    # Mais aussi de renvoyé les oiseaux sur l'ensemble défini s'ils en sortent 
    minx1 = (Simu.mean())-2
    maxx1 = (Simu.mean())+2
    minx2 = (Simu.std())-0.5
    maxx2 = (Simu.std())+0.5
    
    #Voir le PDF pso.pdf pour l'explication mais on peut couper la contrainte de budget d'un agent comme un ensemble de définition d'une fonction
    #Ces valeurs serviront pour l'initiation des oiseaux, on leur définira une valeur aléatoire en -1 et 1 pour leur 1 er vol (1 er itération)
    minvit = -1
    maxvit = 1
    #on travail sur des fonctions à 2 variables mais il est possible d'augmenter le nombre de variable 
    nbvar = 2
    #nbsimulation = 1
    allgb = {}
    allgbp = {}
    #Simul_iteb = np.zeros(shape=(nbsimulation,1)) => pour le machine learning (en cours)
    #Simul_gbest = np.zeros(shape=(nbsimulation,1))=> pour le machine learning (en cours)
    #Simul_xbest = np.zeros(shape=(nbsimulation,nbvar))=> pour le machine learning (en cours)
    #####################################################################################################################################
    #Première itération d'initiation #  
    #for simulation in range(0,nbsimulation,1): => Machine learning 
    inx = np.random.uniform(minx1,maxx1,size=(nbvar, parts)) # on construit nos positions intiales avec un pseudo aléatoire 
    inx[1] = np.random.uniform(minx2,maxx2,size=(nbvar, parts))
    inspeed = np.random.uniform(minvit,maxvit, size=(nbvar, parts)) # Ainsi que nos premières vitesses 
    dfx = pd.DataFrame(inx.T) #on stock nos résultats dans des data frame 
    dfv = pd.DataFrame(inspeed.T)
    #Ici, on calcule les résultats de nos oiseau sur les fonctions choisis
    dfx['F'] = 0  
    for i in range(0,len(dfx),1):
        dfx['F'][i] =np.log(sp.norm.pdf(Simu,dfx[0][i], dfx[1][i])).sum()
    ValueF1 = np.array(dfx['F'])
    #####################################################################################################################################
    #Graphiques
    #Surface : fonction representation => Pour représenter les fonctions sur un graphique 3 d => linspace permet de contruire des surfaces 
    x1 = np.linspace(minx, maxx, 50)
    x2 = np.linspace(minx, maxx, 50)
    z1 = []
    for i in range(0,parts,1):
        z1.append(np.log(sp.norm.pdf(Simu,x1[i],x2[i])).sum())
    Xite = dict([(0,inx.tolist())])
    Yite = dict([(0,ValueF1.tolist())])
    Vite = dict([(0,inspeed.tolist())])
fig = make_subplots(rows=1, cols=2,
                specs=[[{'is_3d': True}, {'is_3d': True}]],
                subplot_titles=['The function representation', 'our birds start their fly'],
                )
fig.add_trace(go.Surface(z=z1),1,1) #La représentation de notre fonction 
fig.add_trace(go.Scatter3d(x=np.array(dfx[0]),y=np.array(dfx[1]),z=np.array(dfx['F']),mode='markers'),1,2) #La répartion de nos oiseaux à l'itération 0
fig.show()
    #####################################################################################################################################
    #Ici, on va définir l'oiseau le plus proche de la solution (avec la valeur la plus faible dans la fonction pour les minimisation et la plus forte pour les maximisations)
    # Gbest est sa valeur dans la fonction et locc le numéro de l'oiseau 
    gbest = max(ValueF1)
    locc = np.argmax(ValueF1)
    #Inertie => vitesse en ligne droite initiale 
    W = wmax - ((wmax-wmin)/maxite)
    #On stock les Gbest pour les résultats de la simulation 
    allgb[0] = gbest
    allgbp[0] = locc
    #En enregistre nos positions initiales et leurs valeurs sur la fonction dans des variables qui vont évoluer dans les boucles 
    Xbest = inx
    ValueF = ValueF1
    #on crée la boucle avec le nb d'itérations: 
    for ite in range(1,maxite,1): #pour l'ensemble du vol 
        for r2 in range(0,parts,1): #pour chaque particules 
            for nbv in range(0,nbvar,1): #pour leurs "2 pattes" => x1 et x2  
                Xbest[nbv,r2] = max(Xbest[nbv,r2],inx[nbv,r2])
                #Voila la fonction de vitesse de nos oiseaux => le premier terme est la vitesse (vit) <=> l'évolution de x1 ou x2 
                #Le second terme est le niveau d'indépendance de l'oiseau (c1) =>  l'oiseau à une "mémoire" et se souviens de toutes ses postions et va aller vers sa valeur la plus basse ou haute 
                #Le troisième terme est le niveau de dépendance de l'oiseau (c2) => Son esprit collaboratif au sein des autres oiseaux, il ira vers le gbest plus rapidement si c2 est élévé 
                # c1 et c2 sont comme des vecteurs de directions et W est l'inertie c'est à dire la vitesse / distance parcouru en cette direction 
                inspeed[nbv,r2] = W*inspeed[nbv][r2] + c1*random.random()*(Xbest[nbv][r2]-inx[nbv][r2]) +c2*random.random()*(inx[nbv][locc]-inx[nbv][r2])
                #Ici, on impose les contraintes aux oiseaux, s'ils sortent de la fonctions ils sont automatiquement ramenés dans la fonction et son ensemble
                if inx[nbv,r2]+ inspeed[nbv,r2] <= minx or inx[nbv,r2]+ inspeed[nbv,r2] >= maxx : 
                    inx[nbv,r2] = random.uniform(maxx,minx)
                else : 
                    inx[nbv,r2] = inx[nbv,r2] + inspeed[nbv,r2]
        #Xite et Vite sont des json/dictionnaire qui vont stocker tous les valeurs des positions et des valeurs à chaque itération => permet de faire les graphiques interactifs
        Xite[ite] = inx.tolist()
        Vite[ite] = inspeed.tolist()
        # On calcule les valeurs des fonctions à chaque itération 
        for r3 in range(0,parts,1):
            ValueF[r3] = 3*((inx[0,r3])**(1/4))*((inx[1,r3])**(3/4))
        Yite[ite] = ValueF.tolist()
        #On rédéfini l'inertie (elle est décroissante au cours du temps => l'oiseau s'épuise)
        W = wmax - ((wmax-wmin)/maxite)*ite
        if max(Yite[ite]) > gbest : 
            gbest = max(Yite[ite])
            locc = np.argmax(Yite[ite])
        else : 
            gbest = gbest 
            locc = locc
    allgb[ite] = gbest #la valeur de l'oiseau dans la fonction => la solution éphémère 
    allgbp[ite] = locc #on localise le numéro de l'oiseau le plus proche 
    df = {}
    df['Xite'] = Xite
    df['Yite'] = Yite
    df['allgb'] = allgb
    df['allgbp'] = allgbp
 
#TEST avec la première fonction : permet de donner un graphique initial au lancement de l'application 

pso(20,0.3,0.1,0.5,Simu)

pso()
