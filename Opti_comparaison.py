
from config_pso import config_pso
from pso_opti import pso_dictionnaire
from pso_fast import pso_opti
from pso_full_information import pso

# our PSO testing (iteration, particules, simulation) => Let's fixe it at

config = config_pso()

print("                                                                                 ")
print("                Full imformation PSO with dataframes                              ")
print("                                                                                 ")
df_result, best_of_info_df, df_birds , timer_pso_pandas = pso(config["function"], config)
print("                                                                                 ")
print("                                                                                 ")


# trying some optimation on parallelisation and numba

print("                                                                                 ")
print("                Keeping only important information in variable local             ")
print("                                                                                 ")


best_of_info ,timer_variables = pso_opti(config["function"], config)

print("                                                                                 ")
print("                                                                                 ")

print("                                                                                 ")
print("                Keeping only important information in JSON                       ")
print("                                                                                 ")

best_of_info ,timer_json = pso_dictionnaire(config["function"], config)

print("                                                                                 ")
print("                                                                                 ")

print(f" Timer for Pandas no opti : {timer_pso_pandas}'s , Variables local : {timer_variables}'s  and JSON : {timer_json}'s")
