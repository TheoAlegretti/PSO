# Application `Dash` pour visualiser la performance de l'algorithme `PSO`

Le but du projet était de construire en python l'algorithme de `Particule Swarm Optimization` qui est un algorithme d'optimisation naturel basé sur le comportement des étourneaux.

__J'ai réalisé ce projet seul dans le cadre du projet Python du semestre 1 de deuxième année de l'ENSAE__. 

## Principe

Le principe est rapidement expliqué sur l'application et [plus en détail sur le papier associé](./PSO/pso.pdf) qui est un article de recherche écrit en collaboration avec Baudrin Marie-Lou et Gruarin Serena dans le cadre de mon master 1 à Sorbonne en Econométrie et Statistiques. Cet algorithme semble répondre aux limites de la descente de gradient en localisant les extrema globaux et pas seulement les extrema locaux.. 

J'ai réalisé cette application sur `Dash`. Le but étant de construire un graphique interactif qui retrace le déplacement des oiseaux dans le temps selon les paramètres utilisés et la fonction a optimiser. On pourra comparer les résultats de la simulation lancée aux résultats théorique de la fonction. L'application est encore améliorable avec du _machine learning_ pour l'optimisation de l'algorithme et avec un effort de mise en page et de fonctions de réponse pour un travail plus soigné. 

## Initialisation 

Pour le lancement de l'application, il vous faudra avoir ces packages installés sur votre ordinateur ou serveur : 

```python
import random 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import math as mt 
import plotly.io as pio
```

Il vous suffit de lancer depuis le terminal de votre ordinateur le script `app.py` et de récupérer l'adresse donnnée par `Dash` (ex :http://127.0.0.1:8050/). Il se peut que l'ouverture abusive du serveur bloque son accès : pas de panique, il vous suffit de _kill_ vos scripts python lancés et de fermer la fenetre de votre navigateur avec le serveur bloqué. Pour kill vos scripts python sur mac ou linux (bash) c'est `pkill -p python`. Pour `Windows`, vous pouvez faire `CTRL+Z` ou `quit()`. 

## Bonus 

(J'ai rajouté un petit bonus sur le script `cryp.py` qui est un système d'encodage de phrase en utilisant comme clé de transformation la suite des 26 premiers nombres premiers combiné au chiffre de César.) 

## En cas de problème

Si vous rencontrez un soucis dans le lancement ou si vous souhaitez discuter du projet, je vous laisse mon mail ci dessous : 

theo.lorthios@ensae.fr
