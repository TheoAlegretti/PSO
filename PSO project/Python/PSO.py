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

def pso(fct) : 
    global Xite,Yite,gbest,inspeed,nbvar,parts,maxx,minx,Vite,maxite
    c1 = 0
    c2 = 0.511
    wmax = 0.75
    wmin = 0.1
    parts = 20
    maxite = 50
    minx = -500
    maxx = 500
    minvit = -1
    maxvit = 1
    nbvar = 2
    nbsimulation = 1
    Simul_iteb = np.zeros(shape=(nbsimulation,1))
    Simul_gbest = np.zeros(shape=(nbsimulation,1))
    Simul_xbest = np.zeros(shape=(nbsimulation,nbvar))
    #Première itération d'initiation # 
    #for simulation in range(0,nbsimulation,1):
    inx = np.random.uniform(minx,maxx,size=(nbvar, parts)) # on construit nos positions intiales avec un pseudo aléatoire 
    inspeed = np.random.uniform(minvit,maxvit, size=(nbvar, parts)) # Ainsi que nos premières vitesses 
    dfx = pd.DataFrame(inx.T)
    dfv = pd.DataFrame(inspeed.T)
    dfx['F'] = 418.9829*2 - dfx[0]*np.sin(abs(dfx[0])**0.5) - dfx[1]*np.sin(abs(dfx[1])**0.5)
    ValueF1 = np.array(dfx['F'])
    #fig = go.Figure(px.scatter_3d(dfx,x=0,y=1,z='F',title="Initial repartition of our birds")) 
    #Surface : fonction representation
    x1 = np.linspace(minx, maxx, 50)
    x2 = np.linspace(minx, maxx, 50)
    v_x1, v_x2 = np.meshgrid(x1, x2)
    z1 = 418.9829*nbvar - x1*np.sin(abs(v_x1)**0.5) - x2*np.sin(abs(v_x2)**0.5)
    Xite = dict([(0,inx.tolist())])
    Yite = dict([(0,ValueF1.tolist())])
    Vite = dict([(0,inspeed.tolist())])
    #fig = go.Figure(go.Surface(z=z1))
    #fig.show()
    #fig = go.Figure(px.scatter_3d(dfx,x=0,y=1,z='F',title="Initial repartition of our birds"))
    #fig.show()
    #fig = go.Figure(go.Scatter3d(x=np.array(dfx[0]),y=np.array(dfx[1]),z=np.array(dfx['F']),mode='markers'))
    #work ok 
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{'is_3d': True}, {'is_3d': True}]],
                        subplot_titles=['The function representation', 'our birds start their fly'],
                        )
    fig.add_trace(go.Surface(z=z1),1,1)
    fig.add_trace(go.Scatter3d(x=np.array(dfx[0]),y=np.array(dfx[1]),z=np.array(dfx['F']),mode='markers'),1,2)
    #fig.add_trace(px.scatter(x=np.array(dfv[0]),y=np.array(dfv[1])),2,1)
    #fig = go.Figure(go.Surface(z=z1))
    #fig.show()
    #fig = go.Figure(px.scatter_3d(dfx,x=0,y=1,z='F',title="Initial repartition of our birds"))
    #fig.show()
    #fig.add_scatter3d(dfx,x=0,y=1,z='F',title="Initial repartition of our birds")
    #fig.show()
    #fonction(inx[RVin,0]**2 + inx[RVin,1]**4 + 2)
    gbest = min(ValueF1)
    locc = np.argmin(ValueF1)
    W = wmax - ((wmax-wmin)/maxite)
    Xbest = inx
    ValueF = ValueF1
    iteb=0
    #on crée la boucle avec le nb d'itérations: 
    for ite in range(1,maxite,1):
        for r2 in range(0,parts,1):
            for nbv in range(0,nbvar,1): 
                Xbest[nbv,r2] = min(Xbest[nbv,r2],inx[nbv,r2])
                #inspeed[nbv,r2] = W*inspeed[nbv][r2] + c2*random.random()*(inx[nbv][locc]-inx[nbv][r2])
                inspeed[nbv,r2] = W*inspeed[nbv][r2] + c1*random.random()*(Xbest[nbv][r2]-inx[nbv][r2]) + c2*random.random()*(inx[nbv][locc]-inx[nbv][r2])
                inx[nbv,r2] = inx[nbv,r2] + inspeed[nbv,r2]
                if inx[nbv,r2] <= minx or inx[nbv,r2] >= maxx : 
                    inx[nbv,r2] = random.uniform(maxx,minx)
                else : 
                    inx[nbv,r2] = inx[nbv,r2] + inspeed[nbv,r2]
        Xite[ite] = inx.tolist()
        Vite[ite] = inspeed.tolist()
        for r3 in range(0,parts,1):
            if fct == 
                ValueF[r3] = 418.9829*nbvar - inx[0,r3]*np.sin(abs(inx[0,r3])**0.5) - inx[1,r3]*np.sin(abs(inx[1,r3])**0.5)
        Yite[ite] = ValueF.tolist()
        W = wmax - ((wmax-wmin)/maxite)*ite
        if min(ValueF) <= gbest : 
            gbest = min(ValueF)
            locc = np.argmin(ValueF)
            iteb = ite
        else :
            gbest = gbest 
            locc = locc

pso()


    #TEST API 

app = dash.Dash(__name__)


app.layout = html.Div([
    html.H1(children='Particule Swarm Optimization'),
    html.Div(children='''
        Cette application interactive permet une visualisation du PSO sur plusieurs fonctions qui semble complexe à résoudre analytiquement ou avec une multitude d'extrema locaux.
        Vous pourrez ainsi jouer avec les paramètres de l'algorithme afin de comprendre ce comportement ontologique aussi bien dans les mathématiques que dans la nature.
    '''),
    html.Label('Dropdown'),
        dcc.Dropdown(
            options=[
                {'label': 'Cobb-Douglas function', 'value': 'COBB'},
                {'label': 'Schwefel function', 'value': 'Schw'},
                {'label': 'Rastrigin function', 'value': 'Ratr'},
                {'label': 'Rosenbrock function', 'value': 'Rosb'}
                {'label': 'Easom function', 'value': 'Eas'}
            ],
            value='COBB'
        ),
    dcc.Graph(id="scatter-plot"),
    html.P("iteration :"),
    dcc.Slider(
        id='my-slider',
        min=0,
        max=maxite,
        step=1,
        value =1,
    )
])  


@app.callback(Output("scatter-plot", "figure"),[Input("my-slider", "value")])
def graphic(value): 
    x1 = np.linspace(minx, maxx, 50)
    x2 = np.linspace(minx, maxx, 50)
    v_x1, v_x2 = np.meshgrid(x1, x2)
    z1 = 418.9829*nbvar - v_x1*np.sin(abs(v_x1)**0.5) - v_x2*np.sin(abs(v_x2)**0.5)
    #our birds
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{'is_3d': True}, {'is_3d': True}]],
                        subplot_titles=['The function representation', f"Birds in the {value} iteration ! "],
                        shared_xaxes=True)
    fig.add_trace(go.Surface(z=z1),1,1)
    fig.add_trace(go.Scatter3d(x=np.array(Xite[value][0]),y=np.array(Xite[value][1]),z=np.array(Yite[value]),mode='markers'),1,2)
    #fig.update_xaxes(range=[minx, maxx],autorange='False', row=1, col=2)
    #fig.update_yaxes(range=[minx, maxx],autorange='False', row=1, col=2)
    fig.update_layout(height=500)
    #fig.update_xaxes(range=[minx, maxx],autorange='False', row=1, col=2)
    #fig.update_yaxes(range=[minx, maxx],autorange='False', row=1, col=2)
    #fig.show()
    return fig  

if __name__ == '__main__' : 
    app.run_server(debug=True)
    
#ps -ef | grep python
#pkill -9 python