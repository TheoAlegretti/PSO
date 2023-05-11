from functions_to_optimise import * 


def config_pso():

    params = {
        "nb_part" : 50, 
        "vit" : 0.1, 
        "c1" : 0.2, 
        "c2" : 0.2,
        "max_ite" : 5000, 
        "nb_simulation_MC" : 50 , 
        "min_x" : -10, 
        "max_x" : 10, 
        "Dim" : 2, 
        "min_max" : "min", 
        "function" : Rosenbrock,
        }
    
    return params

# Function_test_1(np.array([ 0, 5]))