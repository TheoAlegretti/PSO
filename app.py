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


fctsol = {'Simp':'solution analytique donne x = y = 0 avec z = 2'
        ,'COBB':'solution analytique donne x = 15 , y = 15 avec z = 45',
        'Schw':'solution analytique donne x = y = 420 avec z = 0',
        'Ratr':'solution analytique donne x = y = 0 avec z = 0',
        'Rosb':'solution analytique donne x = y = 1 avec z = 0',
        'Eas':'solution analytique donne x = y = 3.1415 avec z = -1'}


def pso(fct,parts,vit,c1,c2) : 
    global Xite,Yite,gbest,inspeed,nbvar,maxx,minx,Vite,locc,allgb,fctname,allgbp,inx,ValueF,df,maxite
    #c1 = 0 #liberté 
    fctname = fct #la fonction update
    #c2 = 0.575 #dépendance
    #wmax = 0.411 optimisé 
    #wmin = 0.23 optimisé 
    #parts = 30
    #wmax = 0.411
    #wmin = 0.23
    wmax = vit + vit/2
    wmin = vit - vit/2
    maxite = 50 #peut être modifier 
    if fct == 'COBB': 
        minx = 0
        maxx = 15
    elif fct == 'Schw': 
        minx = -500
        maxx = 500
    elif fct == 'Ratr': 
        minx = -5
        maxx = 5
    elif fct == 'Rosb': 
        minx = -10
        maxx = 10
    elif fct == 'Simp' : 
        minx = -20
        maxx = 20
    else : 
        minx = -100
        maxx = 100
    minvit = -1
    maxvit = 1
    nbvar = 2
    #nbsimulation = 1
    allgb = {}
    allgbp = {}
    #Simul_iteb = np.zeros(shape=(nbsimulation,1))
    #Simul_gbest = np.zeros(shape=(nbsimulation,1))
    #Simul_xbest = np.zeros(shape=(nbsimulation,nbvar))
    #Première itération d'initiation # 
    #for simulation in range(0,nbsimulation,1):
    inx = np.random.uniform(minx,maxx,size=(nbvar, parts)) # on construit nos positions intiales avec un pseudo aléatoire 
    inspeed = np.random.uniform(minvit,maxvit, size=(nbvar, parts)) # Ainsi que nos premières vitesses 
    dfx = pd.DataFrame(inx.T)
    dfv = pd.DataFrame(inspeed.T)
    if fct == 'COBB': 
        dfx['F'] = 3*((dfx[0])**(1/4))*((dfx[1])**(3/4))
    elif fct == 'Schw': 
        dfx['F'] = 418.9829*2 - dfx[0]*np.sin(abs(dfx[0])**0.5) - dfx[1]*np.sin(abs(dfx[1])**0.5)
    elif fct == 'Ratr': 
        dfx['F'] = 10*2 + (dfx[0]**2-10*np.cos(2*mt.pi*dfx[0])) + (dfx[1]**2-10*np.cos(2*mt.pi*dfx[1]))
    elif fct == 'Rosb': 
        dfx['F'] =100*((dfx[1]-dfx[0]**2)**2) + (dfx[0]- 1)**2
    elif fct =='Simp' :
        dfx['F'] = dfx[1]**2 + dfx[0]**2 + 2 
    else : 
        dfx['F'] =  -np.cos(dfx[0])*np.cos(dfx[1])*np.exp(-(dfx[0]-mt.pi)**2-(dfx[1]-mt.pi)**2)
    ValueF1 = np.array(dfx['F'])
    #fig = go.Figure(px.scatter_3d(dfx,x=0,y=1,z='F',title="Initial repartition of our birds")) 
    #Surface : fonction representation
    x1 = np.linspace(minx, maxx, 50)
    x2 = np.linspace(minx, maxx, 50)
    v_x1, v_x2 = np.meshgrid(x1, x2)
    if fct == 'COBB': 
        z1 = 3*((v_x1)**(1/4))*((v_x2)**(3/4))
    elif fct == 'Schw': 
        z1 = 418.9829*nbvar - v_x1*np.sin(abs(v_x1)**0.5) - v_x2*np.sin(abs(v_x2)**0.5)
    elif fct == 'Ratr': 
        z1 = 10*nbvar + (v_x1**2-10*np.cos(2*mt.pi*v_x1)) + (v_x2**2-10*np.cos(2*mt.pi*v_x2))
    elif fct == 'Rosb': 
        z1 = 100*((v_x2-v_x1**2)**2) + (v_x1-1)**2
    elif fct == 'Simp' : 
        z1 = v_x2**2 + v_x1**2 + 2 
    else : 
        z1 = -np.cos(v_x1)*np.cos(v_x2)*np.exp(-(v_x1-mt.pi)**2-(v_x2-mt.pi)**2)
    #z1 = 418.9829*nbvar - x1*np.sin(abs(v_x1)**0.5) - x2*np.sin(abs(v_x2)**0.5)
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
    if fct == 'COBB' : 
        gbest = max(ValueF1)
        locc = np.argmax(ValueF1)
    else : 
        gbest = min(ValueF1)
        locc = np.argmin(ValueF1)
    W = wmax - ((wmax-wmin)/maxite)
    allgb[0] = gbest
    allgbp[0] = locc
    Xbest = inx
    ValueF = ValueF1
    #on crée la boucle avec le nb d'itérations: 
    for ite in range(1,maxite,1):
        for r2 in range(0,parts,1):
            for nbv in range(0,nbvar,1): 
                if fct == 'COBB': 
                    Xbest[nbv,r2] = max(Xbest[nbv,r2],inx[nbv,r2])
                elif fct == 'Schw': 
                    Xbest[nbv,r2] = min(Xbest[nbv,r2],inx[nbv,r2])
                elif fct == 'Ratr': 
                    Xbest[nbv,r2] = min(Xbest[nbv,r2],inx[nbv,r2])
                elif fct == 'Rosb': 
                    Xbest[nbv,r2] = min(Xbest[nbv,r2],inx[nbv,r2])
                elif fct == 'Simp' : 
                    Xbest[nbv,r2] = min(Xbest[nbv,r2],inx[nbv,r2])
                else : 
                    Xbest[nbv,r2] = min(Xbest[nbv,r2],inx[nbv,r2])
                #Xbest[nbv,r2] = min(Xbest[nbv,r2],inx[nbv,r2])
                #inspeed[nbv,r2] = W*inspeed[nbv][r2] + c2*random.random()*(inx[nbv][locc]-inx[nbv][r2])
                inspeed[nbv,r2] = W*inspeed[nbv][r2] + c1*random.random()*(Xbest[nbv][r2]-inx[nbv][r2]) + c2*random.random()*(inx[nbv][locc]-inx[nbv][r2])
                #inx[nbv,r2] = inx[nbv,r2] + inspeed[nbv,r2]
                if inx[nbv,r2]+ inspeed[nbv,r2] <= minx or inx[nbv,r2]+ inspeed[nbv,r2] >= maxx : 
                    inx[nbv,r2] = random.uniform(maxx,minx)
                else : 
                    inx[nbv,r2] = inx[nbv,r2] + inspeed[nbv,r2]
        Xite[ite] = inx.tolist()
        Vite[ite] = inspeed.tolist()
        for r3 in range(0,parts,1):
            if fct == 'COBB': 
                ValueF[r3] = 3*((inx[0,r3])**(1/4))*((inx[1,r3])**(3/4))
            elif fct == 'Schw': 
                ValueF[r3] = 418.9829*nbvar - inx[0,r3]*np.sin(abs(inx[0,r3])**0.5) - inx[1,r3]*np.sin(abs(inx[1,r3])**0.5)
            elif fct == 'Ratr': 
                ValueF[r3] = 10*nbvar + (inx[0,r3]**2-10*np.cos(2*mt.pi*inx[0,r3])) + (inx[1,r3]**2-10*np.cos(2*mt.pi*inx[1,r3]))
            elif fct == 'Rosb': 
                ValueF[r3] = 100*((inx[1,r3]-inx[0,r3]**2)**2) + (inx[0,r3]- 1)**2
            elif fct =='Simp': 
                ValueF[r3] = inx[1,r3]**2 + inx[0,r3]**2 + 2 
            else : 
                ValueF[r3] = -np.cos(inx[0,r3])*np.cos(inx[1,r3])*np.exp(-(inx[0,r3]-mt.pi)**2-(inx[1,r3]-mt.pi)**2)
            #ValueF[r3] = 418.9829*nbvar - inx[0,r3]*np.sin(abs(inx[0,r3])**0.5) - inx[1,r3]*np.sin(abs(inx[1,r3])**0.5)
        Yite[ite] = ValueF.tolist()
        W = wmax - ((wmax-wmin)/maxite)*ite
        if min(Yite[ite]) < gbest : 
            if fct == 'Schw': 
                gbest = min(Yite[ite])
                locc = np.argmin(Yite[ite])
            elif fct == 'Ratr': 
                gbest = min(Yite[ite])
                locc = np.argmin(Yite[ite])
            elif fct == 'Rosb': 
                gbest = min(Yite[ite])
                locc = np.argmin(Yite[ite])
            elif fct == 'Simp': 
                gbest = min(Yite[ite])
                locc = np.argmin(Yite[ite])
            else : 
                gbest = min(Yite[ite])
                locc = np.argmin(Yite[ite])
        else :
            gbest= gbest
            locc = locc
        if fct == 'COBB': 
            if max(Yite[ite]) > gbest : 
                gbest = max(Yite[ite])
                locc = np.argmax(Yite[ite])
            else : 
                gbest = gbest 
                locc = locc
        allgb[ite] = gbest #la valeur de l'oiseau dans la fonction => la solution éphémère 
        allgbp[ite] = locc #on localise le numéro de l'oiseau le plus proche 
    df = {}
    df['Xite'] = Xite
    df['Yite'] = Yite
    df['allgb'] = allgb
    df['allgbp'] = allgbp
 
#TEST 

pso('Simp',20,0.3,0,0.5)

app = dash.Dash(__name__)




app.layout = html.Div([

    # first row
    html.Div(children=[
        html.H1(html.A(
                'The Particule Swarm Optimisation',
                style={
                    'text-decoration': 'none',
                    'color': 'inherit',
                    'textAlign':'center'
                }
            ))
        ,html.Br(), 
        dcc.Markdown('''
        Cette application interactive permet une visualisation du [PSO](https://www.linkedin.com/feed/update/urn:li:activity:6762648696573186050/) sur plusieurs fonctions qui semble complexe à résoudre analytiquement ou avec une multitude d'extrema locaux (on trouvera leur expression en format latex dans le pdf de l'hyperlien). 
        Vous pourrez ainsi jouer avec les paramètres de l'algorithme afin de comprendre ce comportement ontologique aussi bien dans les mathématiques que dans la nature. Vous pouvez retrouver un devoir de recherche écrit sur la théorie et l'optimisation de cette algorithme. Pour faire simple, on admet un mouvement d'indépendance qui est la liberté de l'oiseau (c1) et un mouvement de dépendance (c2) qui force l'oiseau à aller vers le "gbest", la meilleure solution du problème à l'itération t. Les oiseaux vont parcourir de manière plus intelligentes que la descente de gradien en communiquant entre eux. Il est possible d'améliorer les performances du modèle avec un système de machine learning : "Zoom Effect" : c'est une boucle qui contraint l'ensemble de définition de la fonction et qui compare les résultats entre chaque ensembles. La fonction Rastrigin semble capricieuse avec nos paramètres de bases optimisés : à vous de trouver la bonne paramétrisation pour retrouver la solution théorique !'''),html.Br(),html.Br()
    ], className='row',),
        # first column of second
        html.Div(children=[
            html.Div(children=[
                html.H3(html.A("Paramétrisation : ")), 
                html.P("Itération (mouvements des oiseaux dans le temps)  : "),
                dcc.Slider(
                    id='crossfilter-ite',
                    min=0,
                    max=50,
                    step=1,
                    value =0,
                ),
                   html.Br(),
                html.P("Nombre d'oiseaux (particules) : "),
                dcc.Slider(
                    id='crossfilter-parts',
                    min=0,
                    max=50,
                    step=1,
                    value =20,
                ),
                   html.Br(),
                html.P("Vitesse de l'oiseau en ligne droite : "),
                dcc.Slider(
                    id='crossfilter-vit',
                    min=0,
                    max=2,
                    step=0.1,
                    value =0.3,
                ),
                   html.Br(),
                html.P("Degrés d'indépendance de l'oiseau (c1) : "),
                dcc.Slider(
                    id='crossfilter-c1',
                    min=0,
                    max=2,
                    step=0.1,
                    value =0,
                ),   html.Br(),
                html.P("Degré de dépendance de l'oiseau (c2) : "),
                dcc.Slider(
                    id='crossfilter-c2',
                    min=0,
                    max=2,
                    step=0.1,
                    value =0.5,
                )
                    ], style={'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '0vw', 'margin-top': '0vw','width': '30%'}),

        # second column of second row
                html.Div(children=[
                    html.Label('Fonction disponible : '),
                dcc.Dropdown(id = 'crossfilter-xaxis-column', 
                options=[
                    {'label': 'simple additive fonction', 'value': 'Simp'},
                    {'label': 'Cobb-Douglas function', 'value': 'COBB'},
                    {'label': 'Schwefel function', 'value': 'Schw'},
                    {'label': 'Rastrigin function', 'value': 'Ratr'},
                    {'label': 'Rosenbrock function', 'value': 'Rosb'},
                    {'label': 'Easom function', 'value': 'Eas'}
                ],
                placeholder="Select a fonction to optimize", 
                value = 'Simp'
                    ),
                        dcc.Graph(id="scatter-plot"),

                ], style={'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '0vw', 'margin-top': '0vw','width': '70%'}),
                    ], className='row'),

    # third row
    html.Div(children=[
        html.H3(html.A(
                'Résultat de cette simulation',
                style={
                    'text-decoration': 'none',
                    'color': 'inherit',
                    'textAlign':'center'
                }
            )), 
        html.Div(id='fction'),
        html.Br(),
        html.H3(html.A(
                "Résultat théorique de l'optimisation : ",
                style={
                    'text-decoration': 'none',
                    'color': 'inherit',
                    'textAlign':'center'
                }
            )),
        html.Div(id='fctionsol'),
        html.Br(),
        html.Br()
    ],className='row')
])






@app.callback(Output('fction','children'),[Input("crossfilter-xaxis-column","value"),Input('crossfilter-parts','value'),Input('crossfilter-vit','value'),Input('crossfilter-c1','value'),Input('crossfilter-c2','value')])

def update_pso(xaxis_column_name,parts,vit,c1,c2): 
    pso(xaxis_column_name,parts,vit,c1,c2)
    if xaxis_column_name == 'COBB': 
        bestsol =df['allgb'][max(df['allgb'].keys(), key=(lambda k: df['allgb'][k]))]
        loc = max(df['allgb'].keys(), key=(lambda k: df['allgb'][k]))
        xs = [Xite[loc][0][df['allgbp'][loc]],Xite[loc][1][df['allgbp'][loc]]]
    else : 
        bestsol = df['allgb'][min(df['allgb'].keys(), key=(lambda k: df['allgb'][k]))]
        loc = min(df['allgb'].keys(), key=(lambda k: df['allgb'][k]))
        xs = [Xite[loc][0][df['allgbp'][loc]],Xite[loc][1][df['allgbp'][loc]]]
    return f"Cette simulation nous donne comme solution : {bestsol}   , à l'itération : {loc} , avec les valeurs [x1,x2] : {xs}. "

@app.callback(Output('fctionsol','children'),[Input("crossfilter-xaxis-column","value")])

def update_solfct(xaxis_column_name): 
    return f'Avec cette fonction , on trouve en théorie : {fctsol[xaxis_column_name]}'
    

@app.callback(Output("scatter-plot", "figure"),[Input("crossfilter-ite",component_property= "value"),Input("crossfilter-xaxis-column","value")])

def graphic(ite,xaxis_column_name): 
    #pso(xaxis_column_name)
    fctname = xaxis_column_name
    value = ite
    x1 = np.linspace(minx, maxx, 50)
    x2 = np.linspace(minx, maxx, 50)
    v_x1, v_x2 = np.meshgrid(x1, x2)
    if fctname == 'COBB': 
        z1 = 3*((v_x1)**(1/3))*((v_x2)**(3/4))
        maxz = 15
        minz = 60
    elif fctname == 'Schw': 
        z1 = 418.9829*nbvar - v_x1*np.sin(abs(v_x1)**0.5) - v_x2*np.sin(abs(v_x2)**0.5)
        maxz = 1800
        minz = -5 
    elif fctname == 'Ratr': 
        z1 = 10*nbvar + (v_x1**2-10*np.cos(2*mt.pi*v_x1)) + (v_x2**2-10*np.cos(2*mt.pi*v_x2))
        maxz = 90
        minz = -2
    elif fctname == 'Rosb': 
        z1 = 100*((v_x2-v_x1**2)**2)*nbvar + (v_x1-1)**2
        maxz = 18*(10**4)
        minz = -10000
    elif fctname == 'Simp': 
        z1 = v_x2**2 + v_x1**2 + 2 
        maxz = 700
        minz = -1
    else : 
        z1 = -np.cos(v_x1)*np.cos(v_x2)*np.exp(-(v_x1-mt.pi)**2-(v_x2-mt.pi)**2)    #our birds
        maxz = 0.2
        minz = -1.5
    if fctname == 'COBB' : 
        gbest = max(df['Yite'][value])
        locc = np.argmax(df['Yite'][value])
    else : 
        gbest = min(df['Yite'][value])
        locc = np.argmin(df['Yite'][value])
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{'is_3d': True},{'is_3d': True}]],
                        subplot_titles=['La fonction graphiquement :', f"Itération : {value}, Solution : {format(gbest,'.4f')}"],
                        shared_xaxes=True)
    fig.add_trace(go.Surface(z=z1),1,1)
    fig.add_trace(go.Scatter3d(x=np.array(df['Xite'][value][0]),y=np.array(df['Xite'][value][1]),z=np.array(df['Yite'][value]),mode='markers'),1,2)
    #fig.update_xaxes(range=[minx, maxx],autorange='False', row=1, col=2)
    #fig.update_yaxes(range=[minx, maxx],autorange='False', row=1, col=2)
    #fig.update_layout(height=500)
    fig.update_layout(
    xaxis2={'range': [minx, maxx], 'fixedrange': True, 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': minx,'automargin': False},
    yaxis2={'range': [minx, maxx], 'fixedrange': True, 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': maxx, 'automargin': False},
    scene2={
        'aspectmode': 'cube',
        'xaxis': {'range': [minx, maxx], 'rangemode': 'tozero'},
        'yaxis': {'range': [minx, maxx], 'rangemode': 'tozero'},
        'zaxis': {'range': [minz, maxz], 'rangemode': 'tozero'},
        'aspectratio': {
            'x': 1,
            'y': 1,
            'z': 1,
        },'annotations': [{'x' : np.array(df['Xite'][value][0][locc]),'y' : np.array(df['Xite'][value][1][locc]),'z':np.array(df['Yite'][value][locc]),'text':"Gbest"}]},
    autosize=False,
    height=500)
    #fig.update_xaxes(range=[minx, maxx],autorange='False', row=1, col=2)
    #fig.update_yaxes(range=[minx, maxx],autorange='False', row=1, col=2)
    #fig.show()
    return fig



if __name__ == '__main__' : 
    app.run_server(debug=True)
    
#ps -ef | grep python
#pkill -9 python