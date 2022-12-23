import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math as mt 
import random 
import pandas as pd 
from functions_to_optimise import *




params = {
    "nb_part" : 10, 
    "vit" : 0.3, 
    "c1" : 0.5, 
    "c2" : 0.5,
    "max_ite" : 50, 
    "nb_simulation_MC" : 5 , 
    "min_x" : 0, 
    "max_x" : 15, 
    "Dim" : 2, 
    "min_max" : "max", 
    }



# def pso(fct,params) : 
    # global Xite,Yite,gbest,inspeed,nbvar,maxx,minx,Vite,locc,allgb,fctname,allgbp,inx,ValueF,df,maxite
vit = params['vit']
wmax = vit + vit/2
wmin = vit - vit/2
# if fct == 'COBB': 
#     minx = 0
#     maxx = 15 #Voir le PDF pso.pdf pour l'explication mais on peut couper la contrainte de budget d'un agent comme un ensemble de définition d'une fonction
# elif fct == 'Schw': 
#     minx = -500
#     maxx = 500
# elif fct == 'Ratr': 
#     minx = -5
#     maxx = 5
# elif fct == 'Rosb': 
#     minx = -10
#     maxx = 10
# elif fct == 'Simp' : 
#     minx = -20
#     maxx = 20
# else : 
#     minx = -100
#     maxx = 100

#Ces valeurs serviront pour l'initiation des oiseaux, on leur définira une valeur aléatoire en -1 et 1 pour leur 1 er vol (1 er itération)
minvit = -1
maxvit = 1

#####################################################################################################################################
# On construit les dictionnaires d'itérations (pour chaque vols des oiseaux) et de simulations pour l'estimation par simulation 

#Première itération d'initiation #  


birds  = {
            "simulation" : 
                { 
                0 : 
                    {
                    "positions" : 
                                {
                                    0 : None
                                },
                    "vitesses" : 
                                {
                                    0 : None
                                }
                    }
                }
        }


results  = {
            "simulation" : 
                { 
                0 : 
                    {
                    "output" : 
                                {
                                    0 : None
                                },
                    "best_bird" : 
                                {
                                    0 : None
                                }
                    }
                }
        }




def arg_min_max(array, min_max): 
    """
    This function will found the bird with the max or the min value of the function to optimise
    It will return the index of the bird

    Args:
        array (np.array): The array where we will found the output of the function
        min_max (str): Did we maximise or minimise the function ? 

    Returns:
        int : The index of the best bird 
    """
    if min_max == 'max' : 
        return np.argmax(array)
    elif  min_max == 'min' :
        return np.argmax(array)
    else : 
        print('please, define if you want to maximise or minimise your function on the params dict')



#####################################################################################################################################


# #Graphiques
# #Surface : fonction representation => Pour représenter les fonctions sur un graphique 3 d => linspace permet de contruire des surfaces 
# x1 = np.linspace(minx, maxx, 50)
# x2 = np.linspace(minx, maxx, 50)
# v_x1, v_x2 = np.meshgrid(x1, x2)
# if fct == 'COBB': 
#     z1 = 3*((v_x1)**(1/4))*((v_x2)**(3/4))
# elif fct == 'Schw': 
#     z1 = 418.9829*nbvar - v_x1*np.sin(abs(v_x1)**0.5) - v_x2*np.sin(abs(v_x2)**0.5)
# elif fct == 'Ratr': 
#     z1 = 10*nbvar + (v_x1**2-10*np.cos(2*mt.pi*v_x1)) + (v_x2**2-10*np.cos(2*mt.pi*v_x2))
# elif fct == 'Rosb': 
#     z1 = 100*((v_x2-v_x1**2)**2) + (v_x1-1)**2
# elif fct == 'Simp' : 
#     z1 = v_x2**2 + v_x1**2 + 2 
# else : 
#     z1 = -np.cos(v_x1)*np.cos(v_x2)*np.exp(-(v_x1-mt.pi)**2-(v_x2-mt.pi)**2)
# Xite = dict([(0,inx.tolist())])
# Yite = dict([(0,ValueF1.tolist())])
# Vite = dict([(0,inspeed.tolist())])
# fig = make_subplots(rows=1, cols=2,
#                     specs=[[{'is_3d': True}, {'is_3d': True}]],
#                     subplot_titles=['The function representation', 'our birds start their fly'],
#                     )
# fig.add_trace(go.Surface(z=z1),1,1) #La représentation de notre fonction 
# fig.add_trace(go.Scatter3d(x=np.array(dfx[0]),y=np.array(dfx[1]),z=np.array(dfx['F']),mode='markers'),1,2) #La répartion de nos oiseaux à l'itération 0
# #fig.show => permet de faire une première visualisation 

#####################################################################################################################################
#Ici, on va définir l'oiseau le plus proche de la solution (avec la valeur la plus faible dans la fonction pour les minimisation et la plus forte pour les maximisations)
# Gbest est sa valeur dans la fonction et locc le numéro de l'oiseau 

def actualisation_vitesse(iteration,simu) : 
    global results, birds
    W = wmax - ((wmax-wmin)/iteration)
    for var in range(0,params['Dim']) :  
        birds['simulation'][simu]['vitesses'][iteration][var] = W*birds['simulation'][simu]['vitesses'][iteration-1][var] + params['c1']*random.random()*(birds['simulation'][simu]['positions'][iteration-1][var][results['simulation'][simu]['best_bird'][iteration-1]]-birds['simulation'][simu]['positions'][iteration-1][var]) +params['c2']*random.random()*(birds['simulation'][simu]['positions'][iteration-1][var]-birds['simulation'][simu]['positions'][iteration-1][var])



#Inertie => vitesse en ligne droite initiale 
W = wmax - ((wmax-wmin)/params['max_ite'])

for simu in range(0,params['nb_simulation_MC']):
    birds['simulation'][simu]['positions'][0] = np.random.uniform(params['min_x'],params['max_x'],size=(params['Dim'],params['nb_part']))
    birds['simulation'][simu]['vitesses'][0] = np.random.uniform(minvit,maxvit,size=(params['Dim'],params['nb_part']))
    results['simulation'][simu]['output'][0] = COBB_DOUGLAS(birds['simulation'][simu]['positions'][0])
    results['simulation'][simu]['best_bird'][0] = np.repeat(False,params['nb_part'])
    results['simulation'][simu]['best_bird'][0][arg_min_max(results['simulation'][simu]['output'][0],params['min_max'])] = True
    for iteration in range(1,params['max_ite']): 
        birds['simulation'][simu]['vitesses'][iteration] = W*birds['simulation'][simu]['vitesses'][iteration-1] + params['c1']*random.random()*(Xbest[nbv][r2]-inx[nbv][r2]) +c2*random.random()*(inx[nbv][locc]-inx[nbv][r2])
        birds['simulation'][simu]['positions'][iteration] = np.random.uniform(params['min_x'],params['max_x'],size=(params['Dim'],params['nb_part']))
        results['simulation'][simu]['output'][iteration] = COBB_DOUGLAS(birds['simulation'][0]['positions'][0])
        results['simulation'][simu]['best_bird'][iteration] = np.repeat(False,params['nb_part'])
        results['simulation'][simu]['best_bird'][0][arg_min_max(results['simulation'][0]['output'][0],params['min_max'])] = True
        
#on crée la boucle avec le nb d'itérations: 

for ite in range(1,maxite,1): #pour l'ensemble du vol 
    for r2 in range(0,parts,1): #pour chaque particules 
        for nbv in range(0,nbvar,1): #pour leurs "2 pattes" => x1 et x2  
            if fct == 'COBB': 
                Xbest[nbv,r2] = max(Xbest[nbv,r2],inx[nbv,r2])
            else : 
                Xbest[nbv,r2] = min(Xbest[nbv,r2],inx[nbv,r2])
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
        if fct == 'COBB': 
            ValueF[r3] = 3*((inx[0,r3])**(1/4))*((inx[1,r3])**(3/4))
        elif fct == 'Schw': 
            ValueF[r3] = 418.9829*nbvar - inx[0,r3]*np.sin(abs(inx[0,r3])**0.5) - inx[1,r3]*np.sin(abs(inx[1,r3])**0.5)
        elif fct == 'Ratr': 
            ValueF[r3] = 10*nbvar + (inx[0,r3]**2-10*np.cos(2*mt.pi*inx[0,r3])) + (inx[1,r3]**2-10*np.cos(2*mt.pi*inx[1,r3]))
        elif fct == 'Rosb': 
            ValueF[r3] = 100*((inx[1,r3]-inx[0,r3]**2)**2) + (inx[0,r3]- 1)**2
        elif fct =='Simp': 
            ValueF[r3] = inx[1,r3]**2 + inx[0,r3]**2 + 2 
        else : 
            ValueF[r3] = -np.cos(inx[0,r3])*np.cos(inx[1,r3])*np.exp(-(inx[0,r3]-mt.pi)**2-(inx[1,r3]-mt.pi)**2)
    Yite[ite] = ValueF.tolist()
    #On rédéfini l'inertie (elle est décroissante au cours du temps => l'oiseau s'épuise)
    W = wmax - ((wmax-wmin)/maxite)*ite
    #On redéfini les gbest à chaque itération 
    if min(Yite[ite]) < gbest : 
        if fct == 'Schw': 
            gbest = min(Yite[ite])
            locc = np.argmin(Yite[ite])
        elif fct == 'Ratr': 
            gbest = min(Yite[ite])
            locc = np.argmin(Yite[ite])
        elif fct == 'Rosb': 
            gbest = min(Yite[ite])
            locc = np.argmin(Yite[ite])
        elif fct == 'Simp': 
            gbest = min(Yite[ite])
            locc = np.argmin(Yite[ite])
        else : 
            gbest = min(Yite[ite])
            locc = np.argmin(Yite[ite])
    else :
        gbest= gbest
        locc = locc
    if fct == 'COBB': 
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
#J'ai tous mis dans un dictionnaire géant => + rapide en boucle pour l'interactibilité des graphiques 
