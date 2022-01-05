# PSO
Projet Python : Application PSO sur fonction complexe 

Le but du projet était de construire en python l'algorithme de Particule Swarm Optimization qui est un algorithme d'optimisation naturel basé sur le comportement des étourneaux. Le principe est rapidement expliqué sur l'application et plus en détail sur le papier associé qui est un article de recherche écrit en collaboration avec Baudrin Marie-Lou et Gruarin Serena dans le cadre de mon master 1 à Sorbonne en Econométrie et Statistiques. Cet algorithme semble répondre aux limites de la descente de gradient en localisant les extrema globaux et pas seulement les extrema locaux.. 

J'ai réalisé cette application sur Dash. Le but étant de construire un graphique interactive qui retrace le déplacement des oiseaux dans le temps selon les paramètres utilisés et la fonction a optimiser. On pourra comparer les résultats de la simulation lancé aux résultats théorique de la fonction. L'application est encore améliorable avec du machine learning pour l'optimisation de l'algorithme et avec un effort de mise en page et de fonctions de réponse pour un travail plus soigné. 

J'ai réalisé ce projet seul dans le cadre du projet Python du semestre 1 de deuxième année de l'ENSAE. 

Pour le lancement de l'application, il vous faudra avoir ces packages installés sur votre ordinateur ou serveur : 

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

Il vous suffit de lancer depuis le terminal de votre ordinateur le script app.py et de récupérer l'adresse donnnée par Dash (ex :http://127.0.0.1:8050/). 

Si vous rencontrez un soucis dans le lancement ou si vous souhaitez discuter du projet, je vous laisse mon mail ci dessous : 

theo.lorthios@ensae.fr

(J'ai rajouté un petit bonus sur le script cryp.py qui est un système d'encodage de phrase en utilisant comme clé de transformation la suite des 26 premiers nombres premiers combiné au chiffre de César.) 