import numpy as np
import random
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading

def actualisation_position(birds,results,iteration,simu,params) -> dict:
    """
    Actualise les positions des particules dans l'optimisation par essaim de particules (PSO).

    Args:
    - iteration (int): Le numéro de l'itération actuelle.
    - simu (int): Le numéro de la simulation actuelle.

    Returns:
    - dict: Un dictionnaire contenant les nouvelles positions des particules pour l'itération actuelle.

    Cette fonction met à jour les positions des particules dans l'essaim en utilisant les vitesses calculées à l'étape
    précédente. Elle utilise les formules de mise à jour des positions dans l'algorithme PSO pour calculer la nouvelle
    position de chaque particule en fonction de sa vitesse actuelle. Les nouvelles positions sont stockées dans la
    variable birds['simulation'][simu]['positions'][iteration] et retournées à la fin de la fonction.
    """
    
    birds['simulation'][simu]['positions'][iteration] = {}
    for var in range(0,params['Dim']) :  
        birds['simulation'][simu]['positions'][iteration][var] = birds['simulation'][simu]['positions'][iteration-1][var] + birds['simulation'][simu]['vitesses'][iteration][var]
    return birds['simulation'][simu]['positions'][iteration]


def actualisation_vitesse(birds,results,iteration,simu,params) -> dict:
    """
    Actualise la vitesse des particules dans l'optimisation par essaim de particules (PSO).

    Args:
    - iteration (int): Le numéro de l'itération actuelle.
    - simu (int): Le numéro de la simulation actuelle.

    Returns:
    - dict: Un dictionnaire contenant les vitesses mises à jour des particules pour l'itération actuelle.

    Cette fonction calcule la nouvelle vitesse pour chaque particule de l'essaim en utilisant l'algorithme PSO.
    Elle utilise les paramètres c1, c2, wmax et wmin pour mettre à jour la vitesse des particules en fonction de leurs
    positions actuelles et de leurs meilleures positions trouvées jusqu'à présent. Les nouvelles vitesses sont stockées
    dans la variable birds['simulation'][simu]['vitesses'][iteration] et retournées à la fin de la fonction.
    """
    vit = params["vit"]
    wmax = vit + vit / 2
    wmin = vit - vit / 2
    W = wmax - ((wmax - wmin) / iteration)
    birds["simulation"][simu]["vitesses"][iteration] = {}
    for var in range(0, params["Dim"]):
        birds["simulation"][simu]["vitesses"][iteration][var] = (
            W * birds["simulation"][simu]["vitesses"][iteration-1][var]
            + params["c1"]
            * random.random()
            * (
                birds["simulation"][simu]["positions"][iteration - 1][var][
                    results["simulation"][simu]["best_bird"][iteration - 1]
                ]
                - birds["simulation"][simu]["positions"][iteration - 1][var]
            )
            + params["c2"]
            * random.random()
            * (
                birds["simulation"][simu]["positions"][iteration - 1][var]
                - birds["simulation"][simu]["positions"][iteration - 1][var]
            )
        )
    return birds["simulation"][simu]["vitesses"][iteration]

def arg_min_max(array, min_max):
    """
    Trouve l'oiseau avec la valeur maximale ou minimale de la fonction à optimiser.
    Retourne l'indice de l'oiseau.

    Args:
        array (np.array): Le tableau où nous cherchons la sortie de la fonction.
        min_max (str): Voulons-nous maximiser ou minimiser la fonction ?

    Returns:
        int: L'indice de l'oiseau optimal.
    """
    if min_max == "max":
        return np.argmax(array)
    elif min_max == "min":
        return np.argmin(array)
    else:
        print(
            "please, define if you want to maximise or minimise your function on the params dict"
        )

def run_simulation(birds,results,simu, params):
    minvit = -1
    maxvit = 1
    fct = params['function']
    birds["simulation"][simu]["positions"][0] = np.random.uniform(
        params["min_x"], params["max_x"], size=(params["Dim"], params["nb_part"])
    )
    birds["simulation"][simu]["vitesses"][0] = np.random.uniform(
        minvit, maxvit, size=(params["Dim"], params["nb_part"])
    )
    results["simulation"][simu]["output"][0] = fct(
        birds["simulation"][simu]["positions"][0]
    )
    results["simulation"][simu]["best_bird"][0] = np.repeat(
        False, params["nb_part"]
    )
    results["simulation"][simu]["best_bird"][0][
        arg_min_max(results["simulation"][simu]["output"][0], params["min_max"])
    ] = True
    birds["simulation"][simu + 1] = {"positions": {}, "vitesses": {}}
    results["simulation"][simu + 1] = {"output": {}, "best_bird": {}}
    # print(f"Simulation n° {simu} done")
    for iteration in range(1, params["max_ite"]):
        birds["simulation"][simu]["vitesses"][iteration] = actualisation_vitesse(
            birds,results,iteration,simu,params
        )
        birds["simulation"][simu]["positions"][iteration] = actualisation_position(
            birds,results,iteration,simu,params
        )
        results["simulation"][simu]["output"][iteration] = fct(
            birds["simulation"][simu]["positions"][iteration]
        )
        results["simulation"][simu]["best_bird"][iteration] = np.repeat(
            False, params["nb_part"]
        )
        results["simulation"][simu]["best_bird"][iteration][
            arg_min_max(
                results["simulation"][simu]["output"][iteration], params["min_max"]
            )
        ] = True
    return birds,results

def store_display(results,birds,params,best_of_info,display):
    # On stock les résultats de l'algorithme ici :
    df_result = pd.DataFrame(results["simulation"]).T
    df_birds = pd.DataFrame(birds["simulation"]).T
    for simu in range(0, params["nb_simulation_MC"]):
        best_of_info["avg"][simu] = np.mean(
            results["simulation"][simu]["output"][params["max_ite"] - 1]
        )
        best_of_info["opti"][simu] = results["simulation"][simu]["output"][
            params["max_ite"] - 1
        ][results["simulation"][simu]["best_bird"][simu]]
        best_of_info["var"][simu] = np.var(
            results["simulation"][simu]["output"][params["max_ite"] - 1]
        )
    best_of_info_df = pd.DataFrame(best_of_info)
    result_ite = arg_min_max(best_of_info["opti"], params["min_max"])
    oiseau_pos = arg_min_max(
        results["simulation"][simu]["output"][result_ite], params["min_max"]
    )
    
    
    inputs = {}
    for x_val in range(0, params["Dim"]):
        inputs[f"x_{x_val}"] = np.round(
            birds["simulation"][result_ite]["positions"][params["max_ite"] - 1][x_val][
                oiseau_pos
            ],
            2,
        )
    if display : 
        # quelques logs :
        print(
            f"La meilleure image obtenue est {np.round(best_of_info['opti'][result_ite],2)}"
        )
        print(
            f"Cette image a été obtenue à la simulation n°{result_ite} avec l'oiseau n° {oiseau_pos} avec les inputs suivants : {inputs}"
        )
    return df_result, best_of_info_df, df_birds

def pso_json_v1(fct, params):
    """
    Execute l'algorithme d'optimisation par essaim de particules (PSO) pour trouver la meilleure solution à un problème d'optimisation.

    Args:
    - fct (function): La fonction à optimiser.
    - params (dict): Les paramètres de configuration pour l'algorithme PSO.

    Returns:
    - df_result (pd.DataFrame): Les résultats de chaque itération de l'algorithme pour chaque simulation.
    - best_of_info_df (pd.DataFrame): Les informations sur la meilleure solution trouvée pour chaque simulation.
    - df_birds (pd.DataFrame): Les positions et vitesses des particules pour chaque itération de chaque simulation.
    - timer (float): Le temps d'exécution de l'algorithme en secondes.
    """
    
    begin = time.time()

    #####################################################################################################################################
    # On construit les dictionnaires d'itérations (pour chaque vols des oiseaux) et de simulations pour l'estimation par simulation

    # Première itération d'initiation #
    birds = {"simulation": {0: {"positions": {0: None}, "vitesses": {0: None}}}}

    results = {"simulation": {0: {"output": {0: None}, "best_bird": {0: None}}}}

    best_of_info = {
        "avg": np.zeros(params["nb_simulation_MC"]),
        "opti": np.zeros(params["nb_simulation_MC"]),
        "var": np.zeros(params["nb_simulation_MC"]),
    }

        
    # Algo chained : TO DO : Mettre des @numba boucle ou opti avec parallélisation
    for simu in range(0, params["nb_simulation_MC"]):
        birds,results = run_simulation(birds,results,simu, params)
        
    df_result, best_of_info_df, df_birds  = store_display(results,birds,params,best_of_info,False)

    print(f"PSO run in {np.round(time.time() - begin,2)}' s")
    
    timer = np.round(time.time() - begin,2)

    return df_result, best_of_info_df, df_birds , timer



def run_simulation_wrapper(args):
    birds, results, simu, params = args
    birds, results = run_simulation(birds, results, simu, params)
    return simu, birds, results

def init_dicts(params):
    birds = {"simulation": {}}
    results = {"simulation": {}}
    for simu in range(params["nb_simulation_MC"]):
        birds["simulation"][simu] = {"positions": {0: None}, "vitesses": {0: None}}
        results["simulation"][simu] = {"output": {0: None}, "best_bird": {0: None}}
    return birds, results


#Multi threading impossible in this setting


#Multi processing 
def pso_json_v3(fct, params):
    """
    Execute l'algorithme d'optimisation par essaim de particules (PSO) pour trouver la meilleure solution à un problème d'optimisation.

    Args:
    - fct (function): La fonction à optimiser.
    - params (dict): Les paramètres de configuration pour l'algorithme PSO.

    Returns:
    - df_result (pd.DataFrame): Les résultats de chaque itération de l'algorithme pour chaque simulation.
    - best_of_info_df (pd.DataFrame): Les informations sur la meilleure solution trouvée pour chaque simulation.
    - df_birds (pd.DataFrame): Les positions et vitesses des particules pour chaque itération de chaque simulation.
    - timer (float): Le temps d'exécution de l'algorithme en secondes.
    """
    
    begin = time.time()

    #####################################################################################################################################
    # On construit les dictionnaires d'itérations (pour chaque vols des oiseaux) et de simulations pour l'estimation par simulation

    # Première itération d'initiation #
    birds, results = init_dicts(params)
    best_of_info = {
        "avg": np.zeros(params["nb_simulation_MC"]),
        "opti": np.zeros(params["nb_simulation_MC"]),
        "var": np.zeros(params["nb_simulation_MC"]),
    }

    with ProcessPoolExecutor() as executor:
        simu_params = [(birds, results, simu, params) for simu in range(params["nb_simulation_MC"])]
        results_list = list(executor.map(run_simulation_wrapper, simu_params))

    for simu, updated_birds, updated_results in results_list:
        birds["simulation"][simu] = updated_birds["simulation"][simu]
        results["simulation"][simu] = updated_results["simulation"][simu]
        
        
    df_result, best_of_info_df, df_birds = store_display(results, birds, params, best_of_info, False)

    print(f"PSO run in {np.round(time.time() - begin, 2)}' s")
    
    timer = np.round(time.time() - begin, 2)

    return df_result, best_of_info_df, df_birds, timer

