
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
import scipy.stats as sp
import plotly.figure_factory as ff


df = pd.read_csv(r'/Users/theoalegretti/Desktop/_VIZ_PRICE_ELASTICITY_PREDICTION_PRICE__202209221916.csv',sep=',')



def graph(data): 
    fig = make_subplots(rows=1, cols=2,subplot_titles=(f"Histogram of OCC rate error ", f"Histogram of price error","Test de "))
    fig.add_trace(go.Histogram(x=data['ERROR_OCC'],histnorm='probability density', name='Error_occ'),row=1,col=1)
    fig.add_trace(go.Histogram(x=data['ERROR_PRICE'],histnorm='probability density', name='Error_price'),row=1,col=2)
    print(len(data))
    fig.show()
    
graph(df.sample(frac = 1))

graph(df[df['ID_OD']==642])

# Test avec une loi normale pour les error OCC sur le 642 

from scipy.stats import norm

data = np.array(df[df.ID_OD==642].ERROR_OCC)

fig = go.Figure(go.Histogram(x=data,histnorm='probability density', name='Error_occ'))
fig.show()


from scipy.stats import norm as ss


def log_vraisemblance(mu,sigma,data):
    log = np.log(ss.pdf(data,mu,sigma)).sum()
    return log



# Test de la stabilité des od et des erreurs 

fig = px.histogram(df, x="ERROR_PRICE", color="ID_OD")
fig.show()


# PSO new 

params = {'c1':0.05,'c2':0.5,'part':30,'vit':0.3,'ite':50,'dim':2}

data = np.array(df[df['ID_OD']==642].ERROR_OCC)




def pso_Monte_Carlo_normal_law(params,data,fct):
    # init 
    particules = np.zeros(shape=(params['part'],params['dim']))
    speed = np.zeros(shape=(params['part'],params['dim']))
    value_fct = np.zeros(shape=(params['part'],1))
    diff_mean = random.uniform(np.quantile(data,0.3),np.quantile(data,0.7))
    diff_std = random.uniform(np.quantile(data,0.2),np.quantile(data,0.8))
    borders_param_mean = [data.mean()-diff_mean,data.mean()+diff_mean]
    borders_param_std = [data.std()-diff_std,data.std()+diff_std]
    fig = ff.create_distplot([data], ['distplot'], bin_size=.002)
    fig.show()  
    for p in range(0,len(particules),1): 
        particules[p,0] = random.uniform(borders_param_mean[0],borders_param_mean[1])
        particules[p,1] =  random.uniform(borders_param_std[0],borders_param_std[1])   
        value_fct[p,0] = fct(data,particules[p,0],particules[p,1])
    gbest = min(value_fct)
    particule_best = particules[value_fct.argmin(),:]
    # On lance l'algo 
    iteration = {0:{"parts": particules,"val_fct" : value_fct, "gbest":gbest,"particule_best":particule_best}}
    for ite in range(0,params['ite'],1):
        for p in range(0,len(particules),1) : 
            particules[p,0] = params['vit']*inspeed[r2,nbv] + params['c1']*random.random()*(particule_best[0]-inx[r2,nbv]) + params['c2']*random.random()*(inx[locc,nbv]-inx[r2,nbv])
            particules[p,1] =  params['vit']*inspeed[r2,nbv] + params['c1']*random.random()*(particule_best[1]-inx[r2,nbv]) + params['c2']*random.random()*(inx[locc,nbv]-inx[r2,nbv])
            value_fct[p,0] = fct(data,particules[p,0],particules[p,1])

