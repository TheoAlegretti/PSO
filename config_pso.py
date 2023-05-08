
from functions_to_optimise import * 

def config_pso():

    params = {
        "nb_part" : 200, 
        "vit" : 0.05, 
        "c1" : 0.1, 
        "c2" : 0.6,
        "max_ite" : 500, 
        "nb_simulation_MC" : 500 , 
        "min_x" : -32.768, 
        "max_x" : 32.768, 
        "Dim" : 2, 
        "min_max" : "min", 
        "function" : Ackley,
        }
    
    return params