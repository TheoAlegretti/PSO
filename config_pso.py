from functions.functions_to_optimise import * 


def config_pso():
    """
    Return a config type with all the parameter to test 

    You just have to change the params and rerun the pso_script you prefer 

    Returns:
        dict: with all the hyperparameter,the constraint and the function to optimize 
    """
    
    params = {
        "nb_part" : 50, #Number of birds/ particules in one simulation
        "vit" : 0.1, # The velocity of the bird (will be decrease across the iteration) move it when the support of the function is large
        "c1" : 0.2, # The independance level of the particule from the group (Exploration level) c1 in [0,+inf[
        "c2" : 0.2, # The dependance level of the particule from the group (Exploitation level) c2 in [0,+inf[
        "max_ite" : 5000, # Number of iteration (moves) of each particules in the space in one MC simulation 
        "nb_simulation_MC" : 50 , # The number of Monte Carlo simulation you want to do (to see the convergence)
        "min_x" : -10, # the minimum value that x_i can take, for all i = 1,...,"Dim"
        "max_x" : 10, # the maximum value that x_i can take, for all i = 1,...,"Dim"
        "Dim" : 2, # The number of variable in input if f(x) = y "Dim" = 1, if f(x,z) = y : "Dim" = 2 ... 
        "min_max" : "min",  # "min" if you want to minimize your function else "maximize"
        "function" : Rosenbrock, # change here the function you want to optimize 
        }
    
    return params

