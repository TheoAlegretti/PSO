import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math as mt 
import random 
import pandas as pd 
from functions_to_optimise import *
import time
from config_pso import config_pso


def pso(fct,params) : 
    begin = time.time()
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


    best_of_info = {
        'avg' : np.zeros(params['nb_simulation_MC']) , 
        'opti' : np.zeros(params['nb_simulation_MC']) , 
        'var' : np.zeros(params['nb_simulation_MC'])
    }
    
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
        W = wmax - ((wmax-wmin)/iteration)
        birds['simulation'][simu]['vitesses'][iteration] = {}
        for var in range(0,params['Dim']) :  
            birds['simulation'][simu]['vitesses'][iteration][var] = W*birds['simulation'][simu]['vitesses'][iteration-1][var] + params['c1']*random.random()*(birds['simulation'][simu]['positions'][iteration-1][var][results['simulation'][simu]['best_bird'][iteration-1]]-birds['simulation'][simu]['positions'][iteration-1][var]) +params['c2']*random.random()*(birds['simulation'][simu]['positions'][iteration-1][var]-birds['simulation'][simu]['positions'][iteration-1][var])
        return birds['simulation'][simu]['vitesses'][iteration]

    def actualisation_position(iteration,simu) : 
        birds['simulation'][simu]['positions'][iteration] = {}
        for var in range(0,params['Dim']) :  
            birds['simulation'][simu]['positions'][iteration][var] = birds['simulation'][simu]['positions'][iteration-1][var] + birds['simulation'][simu]['vitesses'][iteration][var]
        return birds['simulation'][simu]['positions'][iteration]
    
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
            return np.argmin(array)
        else : 
            print('please, define if you want to maximise or minimise your function on the params dict')

    for simu in range(0,params['nb_simulation_MC']):
        birds['simulation'][simu]['positions'][0] = np.random.uniform(params['min_x'],params['max_x'],size=(params['Dim'],params['nb_part']))
        birds['simulation'][simu]['vitesses'][0] = np.random.uniform(minvit,maxvit,size=(params['Dim'],params['nb_part']))
        results['simulation'][simu]['output'][0] = fct(birds['simulation'][simu]['positions'][0])
        results['simulation'][simu]['best_bird'][0] = np.repeat(False,params['nb_part'])
        results['simulation'][simu]['best_bird'][0][arg_min_max(results['simulation'][simu]['output'][0],params['min_max'])] = True
        birds['simulation'][simu+1] = {'positions' : {},'vitesses' : {}}
        results['simulation'][simu+1] = {'output' : {},'best_bird' : {}}
        print(f'Simulation n° {simu} done')
        for iteration in range(1,params['max_ite']): 
            birds['simulation'][simu]['vitesses'][iteration] = actualisation_vitesse(iteration,simu)
            birds['simulation'][simu]['positions'][iteration] = actualisation_position(iteration,simu)
            results['simulation'][simu]['output'][iteration] = fct(birds['simulation'][simu]['positions'][iteration])
            results['simulation'][simu]['best_bird'][iteration] = np.repeat(False,params['nb_part'])
            results['simulation'][simu]['best_bird'][iteration][arg_min_max(results['simulation'][simu]['output'][iteration],params['min_max'])] = True  
    # on crée la boucle avec le nb d'itérations: 
    df_result = pd.DataFrame(results['simulation']).T
    df_birds  = pd.DataFrame(birds['simulation']).T
    for simu in range(0,params['nb_simulation_MC']):
        best_of_info['avg'][simu] = np.mean(results['simulation'][simu]['output'][params['max_ite']-1])
        best_of_info['opti'][simu] = results['simulation'][simu]['output'][params['max_ite']-1][results['simulation'][simu]['best_bird'][simu]]
        best_of_info['var'][simu] = np.var(results['simulation'][simu]['output'][params['max_ite']-1])
    best_of_info_df = pd.DataFrame(best_of_info)
    result_ite = arg_min_max(best_of_info['opti'],params['min_max'])
    oiseau_pos = arg_min_max(results['simulation'][simu]['output'][iteration],params['min_max'])
    inputs = {}
    for x_val in range(0,params['Dim']) :
        inputs[f"x_{x_val}"] = np.round(birds['simulation'][result_ite]['positions'][params['max_ite']-1][x_val][oiseau_pos],2)
    print(f"La meilleure image obtenue est {np.round(best_of_info['opti'][result_ite],2)}")
    print(f"Cette image a été obtenue à la simulation n°{result_ite} avec l'oiseau n° {oiseau_pos} avec les inputs suivants : {inputs}")
    print(f"PSO run in {np.round(time.time() - begin,2)}' s")
    return df_result, best_of_info_df ,df_birds

config = config_pso()


pso(config['function'],config)

