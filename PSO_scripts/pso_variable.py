import numpy as np
import random
import time
import concurrent.futures


def actualisation_position(
    fct,
    update_inputs,
    update_output,
    velocity,
    best_bird_input,
    best_bird_output,
    best_personal_inputs,
    best_personal_output,
    params,
):
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

    update_inputs = np.add(update_inputs, velocity)

    # contrainte de minx maxx
    out_range_index = np.where(
        (update_inputs > params["max_x"]) | (update_inputs < params["min_x"])
    )
    update_inputs[:, out_range_index] = np.random.uniform(
        params["min_x"],
        params["max_x"],
        size=(params["Dim"], len(update_inputs[out_range_index])),
    )

    update_output = fct(update_inputs)

    if params["min_max"] == "min":
        update_index = np.where(best_personal_output > update_output)
        best_personal_output = np.minimum(best_personal_output, update_output)
    else:
        update_index = np.where(best_personal_output < update_output)
        best_personal_output = np.maximum(best_personal_output, update_output)

    best_personal_inputs[:, update_index] = update_inputs[:, update_index]

    best_bird_output = best_personal_output[
        arg_min_max(best_personal_output, params["min_max"])
    ]
    best_bird_input = best_personal_inputs[
        :, arg_min_max(best_personal_output, params["min_max"])
    ]

    return (
        update_inputs,
        update_output,
        velocity,
        best_bird_input,
        best_bird_output,
        best_personal_inputs,
        best_personal_output,
    )

# @jit(nopython=True)
def actualisation_vitesse(
    iteration,
    update_inputs,
    velocity,
    best_bird_input,
    best_personal_inputs,
    params,
):
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
    velocity = (
        W * velocity
        + params["c1"]
        * random.random()
        * (np.subtract(update_inputs, best_bird_input.reshape(2, 1)))
        + params["c2"]
        * random.random()
        * (np.subtract(update_inputs, best_personal_inputs))
    )
    return update_inputs, velocity, best_bird_input, best_personal_inputs, iteration

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
    if min_max == "max":
        return np.argmax(array)
    elif min_max == "min":
        return np.argmin(array)
    else:
        print(
            "please, define if you want to maximise or minimise your function on the params dict"
        )

def initialization_pos_vit(fct, params):
    minvit = -1
    maxvit = 1
    update_inputs = np.random.uniform(
        params["min_x"], params["max_x"], size=(params["Dim"], params["nb_part"])
    )
    update_output = fct(update_inputs)
    velocity = np.random.uniform(
        minvit, maxvit, size=(params["Dim"], params["nb_part"])
    )

    best_bird_output = update_output[arg_min_max(update_output, params["min_max"])]
    best_bird_input = update_inputs[
        :, arg_min_max(update_output, params["min_max"])
    ]

    best_personal_output = update_output
    best_personal_inputs = update_inputs

    return (
        update_inputs,
        update_output,
        velocity,
        best_bird_input,
        best_bird_output,
        best_personal_inputs,
        best_personal_output,
    )

def iteration_loop(
    fct,
    params,
):
    (
        update_inputs,
        update_output,
        velocity,
        best_bird_input,
        best_bird_output,
        best_personal_inputs,
        best_personal_output,
    ) = initialization_pos_vit(fct, params)
    iteration = 0
    for iteration in range(1, params["max_ite"]):
        iteration += 1
        (
            update_inputs,
            velocity,
            best_bird_input,
            best_personal_inputs,
            iteration,
        ) = actualisation_vitesse(
            iteration,
            update_inputs,
            velocity,
            best_bird_input,
            best_personal_inputs,
            params,
        )
        (
            update_inputs,
            update_output,
            velocity,
            best_bird_input,
            best_bird_output,
            best_personal_inputs,
            best_personal_output,
        ) = actualisation_position(
            fct,
            update_inputs,
            update_output,
            velocity,
            best_bird_input,
            best_bird_output,
            best_personal_inputs,
            best_personal_output,
            params,
        )
    return best_personal_output, best_bird_input, best_bird_output



def storing_display(display,best_of_info,params):
    best_of_best_index = arg_min_max(best_of_info["opti_out"], params["min_max"])
    # quelques logs :
    if display == True : 
        print(
            "                                                                                                                                                      "
        )

        print(
            "                                                                 *****                                                                                "
        )
        print(
            "                                                                                                                                                      "
        )

        print(
            f"La meilleure image obtenue est {np.round(best_of_info['opti_out'][best_of_best_index],2)}"
        )
        print(
            f"Cette image a été obtenue à la simulation n°{best_of_best_index} avec les inputs suivants : {np.round(best_of_info['opti_in'][best_of_best_index,:],2)}"
        )

        print(
            f"Les oiseaux ont convergé vers {np.round(np.mean(best_of_info['avg']),2)} en moyenne avec une dispersion dans l'espace de {np.round(np.mean(best_of_info['var']),3)}"
        )
        print(
            "                                                                                                                                                      "
        )
    return best_of_info

def pso_var_v1(fct, params):
    begin = time.time()

    #####################################################################################################################################
    # On construit les dictionnaires d'itérations (pour chaque vols des oiseaux) et de simulations pour l'estimation par simulation

    # Première itération d'initiation #

    best_of_info = {
        "avg": np.zeros(params["nb_simulation_MC"]),
        "opti_out": np.zeros(params["nb_simulation_MC"]),
        "opti_in": np.zeros([params["nb_simulation_MC"], params["Dim"]]),
        "var": np.zeros(params["nb_simulation_MC"]),
    }

    for i in range(0,params['nb_simulation_MC']) : 
        best_personal_output, best_bird_input, best_bird_output = iteration_loop(fct,params)
        best_of_info["avg"][i] = np.mean(best_personal_output)
        best_of_info["opti_out"][i] = best_bird_output
        best_of_info["opti_in"][i] = best_bird_input
        best_of_info["var"][i] = np.var(best_personal_output)
    

    best_of_info = storing_display(False,best_of_info,params)
    
    print(f"PSO run in {np.round(time.time() - begin,2)}' s")

    timer = np.round(time.time() - begin, 2)

    return best_of_info, timer


def pso_var_v2(fct, params):
    begin = time.time()

    #####################################################################################################################################
    # On construit les dictionnaires d'itérations (pour chaque vols des oiseaux) et de simulations pour l'estimation par simulation

    # Première itération d'initiation #

    best_of_info = {
        "avg": np.zeros(params["nb_simulation_MC"]),
        "opti_out": np.zeros(params["nb_simulation_MC"]),
        "opti_in": np.zeros([params["nb_simulation_MC"], params["Dim"]]),
        "var": np.zeros(params["nb_simulation_MC"]),
    }


    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(iteration_loop, fct, params)
            for _ in range(params["nb_simulation_MC"])
        ]

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            best_personal_output, best_bird_input, best_bird_output = future.result()
            best_of_info["avg"][i] = np.mean(best_personal_output)
            best_of_info["opti_out"][i] = best_bird_output
            best_of_info["opti_in"][i] = best_bird_input
            best_of_info["var"][i] = np.var(best_personal_output)
            # print(f"Simulation n° {i} done")
    
    
    best_of_info = storing_display(False,best_of_info,params)
    print(f"PSO run in {np.round(time.time() - begin,2)}' s")
    timer = np.round(time.time() - begin, 2)
    return best_of_info, timer


# Multi processing

def pso_var_v3(fct, params):
    begin = time.time()

    #####################################################################################################################################
    # On construit les dictionnaires d'itérations (pour chaque vols des oiseaux) et de simulations pour l'estimation par simulation

    # Première itération d'initiation #

    best_of_info = {
        "avg": np.zeros(params["nb_simulation_MC"]),
        "opti_out": np.zeros(params["nb_simulation_MC"]),
        "opti_in": np.zeros([params["nb_simulation_MC"], params["Dim"]]),
        "var": np.zeros(params["nb_simulation_MC"]),
    }


    with concurrent.futures.ProcessPoolExecutor() as executor: 
        futures = [
            executor.submit(iteration_loop, fct, params)
            for _ in range(params["nb_simulation_MC"])
        ]


        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            best_personal_output, best_bird_input, best_bird_output = future.result()
            best_of_info["avg"][i] = np.mean(best_personal_output)
            best_of_info["opti_out"][i] = best_bird_output
            best_of_info["opti_in"][i] = best_bird_input
            best_of_info["var"][i] = np.var(best_personal_output)
            # print(f"Simulation n° {i} done")
    
    
    best_of_info = storing_display(False,best_of_info,params)
    print(f"PSO run in {np.round(time.time() - begin,2)}' s")
    timer = np.round(time.time() - begin, 2)
    
    return best_of_info, timer