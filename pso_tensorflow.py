import numpy as np
import tensorflow as tf
import time 
from config_pso import config_pso

def pso_opti_tf(fct, params):
    begin = time.time()

    # Convertir les paramètres en tenseurs TensorFlow
    min_x = tf.constant(params["min_x"], dtype=tf.float32)
    max_x = tf.constant(params["max_x"], dtype=tf.float32)
    min_max = params["min_max"]
    nb_part = params["nb_part"]
    max_ite = params["max_ite"]
    c1 = tf.constant(params["c1"], dtype=tf.float32)
    c2 = tf.constant(params["c2"], dtype=tf.float32)

    wmax = params["vit"] + params["vit"] / 2
    wmin = params["vit"] - params["vit"] / 2
    minvit = -1
    maxvit = 1

    # Initialisation des particules et des vitesses
    particles = tf.Variable(np.random.uniform(min_x, max_x, (nb_part, params['Dim'])), dtype=tf.float32)
    velocities = tf.Variable(np.random.uniform(minvit, maxvit, (nb_part, params['Dim'])), dtype=tf.float32)

    # Calcul des coûts initiaux
    costs = fct(particles)
    best_costs = tf.identity(costs)
    best_positions = tf.identity(particles)
    global_best_cost_index = tf.argmin(costs) if min_max == "min" else tf.argmax(costs)
    global_best_cost = tf.gather(costs, global_best_cost_index)
    global_best_position = tf.gather(particles, global_best_cost_index)

    # Boucle d'optimisation
    for iteration in range(1, max_ite):
        r1 = tf.random.uniform((nb_part, params['Dim']))
        r2 = tf.random.uniform((nb_part, params['Dim']))
        W = wmax - ((wmax - wmin) / iteration)
        
        # Mise à jour des vitesses
        cognitive = c1 * r1 * (best_positions - particles)
        social = c2 * r2 * (global_best_position - particles)
        velocities.assign(W * velocities + cognitive + social)

        # Mise à jour des positions
        particles.assign_add(velocities)

        # Vérification des limites
        particles.assign(tf.clip_by_value(particles, min_x, max_x))

        # Mise à jour des coûts
        costs = fct(particles)

        if min_max == "min":
            # Mise à jour des meilleurs coûts personnels
            improved_mask = costs < best_costs
            best_costs = tf.where(improved_mask, costs, best_costs)
            best_positions = tf.where(tf.reshape(improved_mask, (-1, 1)), tf.expand_dims(particles, -1), tf.expand_dims(best_positions, -1))
            best_positions = tf.squeeze(best_positions, -1)

            # Mise à jour du meilleur coût global
            new_global_best_cost_index = tf.argmin(costs)
            new_global_best_cost = tf.gather(costs, new_global_best_cost_index)
            if new_global_best_cost < global_best_cost:
                global_best_cost = new_global_best_cost
                global_best_position = tf.gather(particles, new_global_best_cost_index)

        else:
            # Mise à jour des meilleurs coûts personnels
            improved_mask = costs > best_costs
            best_costs = tf.where(improved_mask, costs, best_costs)
            best_positions = tf.where(tf.reshape(improved_mask, (-1, 1)), tf.expand_dims(particles, -1), tf.expand_dims(best_positions, -1))
            best_positions = tf.squeeze(best_positions, -1)
            # Mise à jour du meilleur coût global
            new_global_best_cost_index = tf.argmax(costs)
            new_global_best_cost = tf.gather(costs, new_global_best_cost_index)
            if new_global_best_cost > global_best_cost:
                global_best_cost = new_global_best_cost
                global_best_position = tf.gather(particles, new_global_best_cost_index)

    # Affichage des résultats
    print(f"Meilleure solution: {global_best_position.numpy()}")
    print(f"Meilleur coût: {global_best_cost.numpy()}")

    timer = np.round(time.time() - begin, 2)
    print(f"PSO run in {timer}' s")

    return global_best_cost, global_best_position, timer

params = config_pso()

pso_opti_tf(params['function'], params)