# Application Dash — Visualisation de l’algorithme **PSO**

Ce projet présente une application Python, développée avec `Dash`, pour visualiser les performances de l'algorithme **Particle Swarm Optimization** (PSO). Cet algorithme d'optimisation est inspiré du comportement collectif des essaims d'étourneaux.

> **Projet réalisé individuellement dans le cadre du projet Python (S2, 2e année ENSAE).**

-----

## Principe

L'algorithme PSO surmonte les limites des méthodes de descente de gradient en explorant l'espace de recherche pour détecter les **extrema globaux**, et non seulement les extrema locaux.

Le principe de l'algorithme est expliqué sur l'application et détaillé dans le [papier associé](https://www.google.com/search?q=./PSO/pso.pdf), co-écrit avec Baudrin Marie-Lou et Gruarin Serena (M1 Sorbonne, Économétrie et Statistiques).

L'application interactive permet de :

  * Visualiser le **déplacement des particules** dans l'espace de recherche en fonction de la fonction à optimiser.
  * Comparer les résultats de la simulation avec les **résultats théoriques**.
  * Tester et modifier les **paramètres** de l'algorithme, et explorer différentes fonctions.

> Des améliorations restent possibles (optimisation des hyperparamètres par *machine learning*, amélioration de l'interface, ajout de nouvelles fonctionnalités).

-----

## Installation & Lancement

### Pré-requis

  * Python 3.x
  * Un environnement virtuel est conseillé (`uv` est recommandé) :

<!-- end list -->

```bash
pip install uv
uv venv
source .venv/bin/activate
uv pip install -r dashbord/requirements.txt
```

### Démarrage

Pour lancer l'application, exécutez la commande suivante :

```bash
python dashbord/app.py
```

Accédez ensuite à l'adresse indiquée (par exemple : `http://127.0.0.1:8050/`).

### Lancer avec Docker (recommandé)

Le service web Dash/Flask et la configuration Docker sont fournis dans le dossier `dashbord`.

1) Construire l'image Docker (depuis la racine du projet) :

```bash
# contexte = dossier dashbord (contient Dockerfile et requirements.txt)
docker build -t pso:latest ./dashbord
```

2) Lancer le conteneur (mappe le port 5002 du conteneur sur le port 5002 de la machine hôte) :

```bash
docker run -d --name pso_container -p 5002:5002 pso:latest
```

3) Vérifier le statut et les logs :

```bash
docker ps --filter "name=pso_container"
docker logs --tail 200 pso_container
# pour suivre les logs en temps réel
docker logs -f pso_container
```

4) Accéder à l'application dans votre navigateur :

http://localhost:5002


### Notebook d'exemples

Le notebook `example_PSO_uses.ipynb` contient une explication détaillée de l'algorithme et des exemples d'utilisation.

### Dépannage

En cas de blocage du serveur Dash, fermez les fenêtres du navigateur concernées et terminez les processus Python :

  * **Mac/Linux** : `pkill -f python`
  * **Windows** : `CTRL+C` ou utilisez `quit()` dans la console Python.

-----

## Utiliser l'algorithme PSO sur vos propres fonctions

### 1\. Définir votre fonction

Voici un exemple avec la fonction Cobb-Douglas :

```python
def COBB_DOUGLAS(input: np.array):
    """
    Exemple de fonction à optimiser : Cobb-Douglas
    f(x1, x2) = 3 * x1**(1/4) * x2**(3/4), pour x1, x2 ∈ [0, 15]
    Solution optimale : x1 = x2 = 15, f* = 45
    """
    return np.array(3 * (input[0])**(1/4) * (input[1])**(3/4))
```

### 2\. Configurer les paramètres

Exemple de configuration des paramètres :

```python
params = {
    "nb_part": 50,               # Nombre de particules
    "vit": 0.1,                  # Vitesse initiale
    "c1": 0.2,                   # Coefficient d'indépendance (exploration)
    "c2": 0.2,                   # Coefficient de dépendance (exploitation)
    "max_ite": 1000,             # Nombre d’itérations maximal
    "nb_simulation_MC": 20,      # Nombre de simulations Monte Carlo
    "min_x": -5.12,
    "max_x": 5.12,
    "Dim": 2,                    # Dimension du problème
    "min_max": "min",            # 'min' pour minimisation ou 'max' pour maximisation
    "function": COBB_DOUGLAS     # Fonction à optimiser
}
```

### 3\. Lancer l’algorithme

```python
from PSO_scripts.pso_JSON import pso_json_v1

df_result, best_of_info_df, df_birds, timer = pso_json_v1(params["function"], params)
```

**Sorties :**

  * `df_result` (`pd.DataFrame`) : Résultats par itération/simulation.
  * `best_of_info_df` (`pd.DataFrame`) : Meilleures solutions trouvées par simulation.
  * `df_birds` (`pd.DataFrame`) : Positions et vitesses des particules.
  * `timer` (`float`) : Temps d’exécution de l'algorithme (en secondes).

-----

## Contact

Pour toute question ou suggestion, n’hésitez pas à me contacter :
theo.lorthios@gmail.com

Bon vol \!

------------------------------------------------------------
# English version 

------------------------------------------------------------

# Dash Application — **PSO** Algorithm Visualization

This project offers a Python application developed with `Dash` to visualize the performance of the **Particle Swarm Optimization** (PSO) algorithm, an optimization algorithm inspired by the collective behavior of starling flocks.

> **This project was individually completed as part of the Python project course (S2, 2nd year ENSAE).**

-----

## Principle

The PSO algorithm overcomes the limitations of gradient descent methods by exploring the search space to detect **global extrema**, not just local ones.

The principle of the algorithm is explained within the application itself and detailed in the [associated paper](https://www.google.com/search?q=./PSO/pso.pdf), co-authored with Baudrin Marie-Lou and Gruarin Serena (M1 Sorbonne, Econometrics and Statistics).

The interactive application allows you to:

  * Visualize the **movement of particles** in the search space according to the function being optimized.
  * Compare simulation results with **theoretical results**.
  * Test and modify **parameters**, and explore different functions.

> Further improvements are possible (hyperparameter optimization via *machine learning*, interface enhancements, new features).

-----

## Installation & Launch

### Prerequisites

  * Python 3.x
  * A virtual environment is recommended (`uv` is preferred):

<!-- end list -->

```bash
pip install uv
uv venv
source .venv/bin/activate
uv pip install -r dashboard/requirements.txt
```

### Starting the Application

To launch the application, run the following command:

```bash
python dashboard/app.py
```

Then, access the indicated address (e.g., `http://127.0.0.1:8050/`).

### Example Notebook

The `example_PSO_uses.ipynb` notebook provides a detailed explanation of the algorithm and usage examples.

### Troubleshooting

If the Dash server gets stuck, close the relevant browser windows and kill the Python processes:

  * **Mac/Linux**: `pkill -f python`
  * **Windows**: `CTRL+C` or use `quit()` in the Python console.

-----

## Using the PSO Algorithm with Your Own Functions

### 1\. Define Your Function

Here's an example using the Cobb-Douglas function:

```python
def COBB_DOUGLAS(input: np.array):
    """
    Example function to optimize: Cobb-Douglas
    f(x1, x2) = 3 * x1**(1/4) * x2**(3/4), for x1, x2 ∈ [0, 15]
    Optimal solution: x1 = x2 = 15, f* = 45
    """
    return np.array(3 * (input[0])**(1/4) * (input[1])**(3/4))
```

### 2\. Configure Parameters

Example parameter configuration:

```python
params = {
    "nb_part": 50,               # Number of particles
    "vit": 0.1,                  # Initial velocity
    "c1": 0.2,                   # Independence coefficient (exploration)
    "c2": 0.2,                   # Dependence coefficient (exploitation)
    "max_ite": 1000,             # Maximum number of iterations
    "nb_simulation_MC": 20,      # Number of Monte Carlo simulations
    "min_x": -5.12,
    "max_x": 5.12,
    "Dim": 2,                    # Problem dimension
    "min_max": "min",            # 'min' for minimization or 'max' for maximization
    "function": COBB_DOUGLAS     # Function to optimize
}
```

### 3\. Run the Algorithm

```python
from PSO_scripts.pso_JSON import pso_json_v1

df_result, best_of_info_df, df_birds, timer = pso_json_v1(params["function"], params)
```

**Outputs:**

  * `df_result` (`pd.DataFrame`): Results per iteration/simulation.
  * `best_of_info_df` (`pd.DataFrame`): Best solutions found per simulation.
  * `df_birds` (`pd.DataFrame`): Particle positions and velocities.
  * `timer` (`float`): Algorithm execution time (in seconds).

-----

## Contact

For any questions or suggestions, feel free to contact me:
theo.lorthios@gmail.com

Happy flying\!