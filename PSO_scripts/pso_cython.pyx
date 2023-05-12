# pso_cython.pyx

import numpy as np
cimport numpy as np
import random
import pandas as pd
from libc.math cimport sqrt
from functions_to_optimise import *

cdef extern from "time.h":
    double time()

cdef double[:] actualisation_position(int iteration, int simu, dict birds, dict params):
    cdef np.ndarray[np.double_t, ndim=1] new_positions = np.empty(params['Dim'], dtype=np.double)
    for var in range(params['Dim']):
        new_positions[var] = birds['simulation'][simu]['positions'][iteration-1][var] + birds['simulation'][simu]['vitesses'][iteration][var]
    return new_positions

cdef double[:] actualisation_vitesse(int iteration, int simu, dict birds, dict params, double wmax, double wmin):
    cdef double W = wmax - ((wmax - wmin) / iteration)
    cdef np.ndarray[np.double_t, ndim=1] new_vitesses = np.empty(params['Dim'], dtype=np.double)
    for var in range(params['Dim']):
        new_vitesses[var] = (
            W * birds["simulation"][simu]["vitesses"][iteration - 1][var]
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
    return new_vitesses

cdef int arg_min_max(np.ndarray[np.double_t, ndim=1] array, str min_max):
    if min_max == "max":
        return np.argmax(array)
    elif min_max == "min":
        return np.argmin(array)
    else:
        raise ValueError("Please define if you want to maximize or minimize your function in the params dict")

cpdef tuple pso(fct, params):
    cdef double begin = time()

    cdef double vit = params["vit"]
    cdef double wmax = vit + vit / 2
    cdef double wmin = vit - vit / 2
    cdef double minvit = -1
    cdef double maxvit = 1

    # (the rest of your code remains unchanged)
    birds = {"simulation": {0: {"positions": {0: None}, "vitesses": {0: None}}}}
    results = {"simulation": {0: {"output": {0: None}, "best_bird": {0: None}}}}
    best_of_info = {
        "avg": np.zeros(params["nb_simulation_MC"]),
        "opti": np.zeros(params["nb_simulation_MC"]),
        "var": np.zeros(params["nb_simulation_MC"]),
    }

    for simu in range(0, params["nb_simulation_MC"]):
        birds["simulation"][simu]["positions"][0] = np.random.uniform(
            params["min_x"], params["max_x"], size=(params["Dim"], params["nb_part"])
        )
        birds["simulation"][simu]["vitesses"][0] = np.random.uniform(
            minvit, maxvit, size=(params["Dim"], params["nb_part
            for iteration in range(1, params["nb_iter"]):
                birds["simulation"][simu]["positions"][iteration] = actualisation_position(iteration, simu, birds, params)
                birds["simulation"][simu]["vitesses"][iteration] = actualisation_vitesse(iteration, simu, birds, params, wmax, wmin)
                results["simulation"][simu]["output"][iteration] = fct(birds["simulation"][simu]["positions"][iteration])

                results["simulation"][simu]["best_bird"][iteration] = arg_min_max(results["simulation"][simu]["output"][iteration], params["min_max"])

                best_of_info["avg"][simu] += results["simulation"][simu]["output"][iteration][results["simulation"][simu]["best_bird"][iteration]] / params["nb_iter"]
                best_of_info["opti"][simu] = min(best_of_info["opti"][simu], results["simulation"][simu]["output"][iteration][results["simulation"][simu]["best_bird"][iteration]])
                best_of_info["var"][simu] += results["simulation"][simu]["output"][iteration][results["simulation"][simu]["best_bird"][iteration]] ** 2 / params["nb_iter"]

    end_time = time() - begin
    return best_of_info, end_time