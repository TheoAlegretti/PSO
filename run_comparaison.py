from config_pso import config_pso
from PSO_scripts.pso_low_memory import pso_LM_v1,pso_LM_v2,pso_LM_v3
from PSO_scripts.pso_variable import pso_var_v1,pso_var_v2,pso_var_v3
from PSO_scripts.pso_JSON import pso_json_v1,pso_json_v3


def main():
    # Your code that calls the pso_LM_v2 function, e.g.:
    config = config_pso()
    
    print('Low_Memory Json naive')
    
    best_of_info, timer_LM_v1 = pso_LM_v1(config["function"], config)
    
    print('Low_Memory Json with multi-threading')
    
    best_of_info, timer_LM_v2 = pso_LM_v2(config["function"], config)
    
    print('Low_Memory Json with multi-processing')
    
    best_of_info, timer_LM_v2 = pso_LM_v3(config["function"], config)

    
    print('Full_Memory Json naive')
    
    df_result, best_of_info_df, df_birds, time_json_v1 = pso_json_v1(config["function"], config)

    print('Full_Memory Json with multiprocessing')
    
    df_result, best_of_info_df, df_birds, time_json_v3 = pso_json_v3(config["function"], config)
    
    print('Low_Memory variable naive')
    
    best_of_info, timer_var_v1 = pso_var_v1(config["function"], config)
    
    print('Low_Memory variable with multi-threading')

    best_of_info, timer_var_v2 = pso_var_v2(config["function"], config)

    print('Low_Memory variable with multi-processing')

    best_of_info, timer_var_v3 = pso_var_v3(config["function"], config)




if __name__ == '__main__':
    main()
