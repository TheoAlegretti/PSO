from update_pso import pso
from opti_pso import *
from config_pso import config_pso
from numpy import *
from numpy.linalg import norm
import time
import numpy as np


# our PSO testing (iteration, particules, simulation) => Let's fixe it at

config = config_pso()

df_result, best_of_info_df, df_birds = pso(config["function"], config)

# trying some optimation on parallelisation and numba

# best_of_info_df, inputs = pso_opti(config["function"], config)
