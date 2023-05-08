import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math as mt 
import random 
import pandas as pd 
from typing import Callable, Dict, Tuple


def pso(fct: Callable, params: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Particle Swarm Optimization Algorithm to optimize a given function.

    Args:
        fct (Callable): The function to optimize.
        params (Dict): The parameters used for the algorithm.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple with dataframes for the results of the simulations, 
            the best results obtained for each simulation and the information about the birds.
    """
    vit = params['vit']
    wmax = vit + vit/2
    wmin = vit - vit/2

    minvit = -1
    maxvit = 1

    # On construit les dictionnaires d'itérations (pour chaque vol des oiseaux) 
    # et de simulations pour l'estimation par simulation
    birds = {"simulation": {0: {"positions": {0: None}, "vitesses": {0: None}}}}
    results = {"simulation": {0: {"output": {0: None}, "best_bird": {0: None}}}}

    best_of_info = {"avg": np.zeros(params["nb_simulation_MC"]), 
                    "opti": np.zeros(params["nb_simulation_MC"]), 
                    "var": np.zeros(params["nb_simulation_MC"])}

    def actualisation_vitesse(iteration: int, simu: int) -> Dict:
        """Update the speed of the birds at a given iteration and simulation.

        Args:
            iteration (int): The iteration number.
            simu (int): The simulation number.

        Returns:
            Dict: The updated speeds of the birds.
        """
        W = wmax - ((wmax-wmin)/iteration)
        birds["simulation"][simu]["vitesses"][iteration] = {}
        for var in range(0, params["Dim"]):
            birds["simulation"][simu]["vitesses"][iteration][var] = (
                W*birds["simulation"][simu]["vitesses"][iteration-1][var] + 
                params["c1"]*random.random()*(birds["simulation"][simu]["positions"][iteration-1][var][results["simulation"][simu]["best_bird"][iteration-1]]-birds["simulation"][simu]["positions"][iteration-1][var]) + 
                params["c2"]*random.random()*(birds["simulation"][simu]["positions"][iteration-1][var]-birds["simulation"][simu]["positions"][iteration-1][var]
            )
        return birds["simulation"][simu]["vitesses"][iteration]
    
    def actualisation_position(iteration, simu):
        """
        This function updates the position of each bird in the swarm for a given iteration and simulation.

        Args:
            iteration (int): The current iteration of the PSO algorithm.
            simu (int): The current simulation of the PSO algorithm.

        Returns:
            dict: A dictionary containing the updated positions for each bird in the swarm.
        """
        birds['simulation'][simu]['positions'][iteration] = {}
        for var in range(0, params['Dim']):
            birds['simulation'][simu]['positions'][iteration][var] = birds['simulation'][simu]['positions'][iteration - 1][var] + birds['simulation'][simu]['vitesses'][iteration][var]
            # Apply constraints on the positions if they go beyond the range
            if birds['simulation'][simu]['positions'][iteration][var] > params['max_x']:
                birds['simulation'][simu]['positions'][iteration][var] = params['max_x']
            elif birds['simulation'][simu]['positions'][iteration][var] < params['min_x']:
                birds['simulation'][simu]['positions'][iteration][var] = params['min_x']
    return birds['simulation'][simu]['positions'][iteration]