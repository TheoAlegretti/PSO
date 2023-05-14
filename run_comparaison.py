from config_pso import config_pso
from PSO_scripts.pso_low_memory import pso_LM_v1,pso_LM_v2,pso_LM_v3
from PSO_scripts.pso_variable import pso_var_v1,pso_var_v2,pso_var_v3
from PSO_scripts.pso_JSON import pso_json_v1,pso_json_v3
import pandas as pd 


def main():
    # Your code that calls the pso_LM_v2 function, e.g.:
    timers_iterations_LM_v1, timers_iterations_LM_v2,timers_iterations_LM_v3,timers_iterations_json_v1,timers_iterations_json_v3,timers_iterations_var_v1,timers_iterations_var_v2 ,timers_iterations_var_v3 = [],[],[],[],[],[],[],[]

    iterations_level = []
    
    perf_iterations_LM_v1, perf_iterations_LM_v2,perf_iterations_LM_v3,perf_iterations_json_v1,perf_iterations_json_v3,perf_iterations_var_v1,perf_iterations_var_v2 ,perf_iterations_var_v3 = [],[],[],[],[],[],[],[]
    
    config = config_pso()

    for ite in range(10,500,10) : 
        config['nb_simulation_MC'] = ite
        iterations_level.append(ite)

        print('Low_Memory Json naive')
        best_of_info, timer_LM_v1 = pso_LM_v1(config["function"], config)
        timers_iterations_LM_v1.append(timer_LM_v1)
        perf_iterations_LM_v1.append(best_of_info['opti_out'].mean())
        
        
        print('Low_Memory Json with multi-threading')
        best_of_info, timer_LM_v2 = pso_LM_v2(config["function"], config)
        timers_iterations_LM_v2.append(timer_LM_v2)
        perf_iterations_LM_v2.append(best_of_info['opti_out'].mean())

        print('Low_Memory Json with multi-processing')
        best_of_info, timer_LM_v3 = pso_LM_v3(config["function"], config)
        perf_iterations_LM_v3.append(best_of_info['opti_out'].mean())
        timers_iterations_LM_v3.append(timer_LM_v3)

        print('Full_Memory Json naive')
        df_result, best_of_info_df, df_birds, time_json_v1 = pso_json_v1(config["function"], config)
        timers_iterations_json_v1.append(time_json_v1)
        perf_iterations_json_v1.append(best_of_info_df.opti.mean())

        print('Full_Memory Json with multiprocessing')
        df_result, best_of_info_df, df_birds, time_json_v3 = pso_json_v3(config["function"], config)
        timers_iterations_json_v3.append(time_json_v3)
        perf_iterations_json_v3.append(best_of_info_df.opti.mean())

        print('Low_Memory variable naive')
        best_of_info, timer_var_v1 = pso_var_v1(config["function"], config)
        timers_iterations_var_v1.append(timer_var_v1)
        perf_iterations_var_v1.append(best_of_info['opti_out'].mean())
        
        print('Low_Memory variable with multi-threading')
        best_of_info, timer_var_v2 = pso_var_v2(config["function"], config)
        timers_iterations_var_v2.append(timer_var_v2)
        perf_iterations_var_v2.append(best_of_info['opti_out'].mean())

        print('Low_Memory variable with multi-processing')
        best_of_info, timer_var_v3 = pso_var_v3(config["function"], config)
        timers_iterations_var_v3.append(timer_var_v3)
        perf_iterations_var_v3.append(best_of_info['opti_out'].mean())
        
        print("Let's increase the number of iteration ! ")
        
    
    df = pd.DataFrame(iterations_level,columns=['nb_part'])
    list_model = ['LM_v1','LM_v2','LM_v3','json_v1','json_v3','var_v1','var_v2','var_v3']
    for model in list_model : 
        df[f'timer_{model}'] = locals()[f'timers_iterations_{model}']
        df[f'perf_{model}'] = locals()[f'perf_iterations_{model}']
    
    df.to_csv('/Users/theoalegretti/Documents/GitHub/PSO/data/monte_carlo_speed_perf.csv')
    print("ok c'est fini")
    




if __name__ == '__main__':
    main()
