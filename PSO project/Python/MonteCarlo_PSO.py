
import random
from turtle import title 
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
from scipy.stats import norm as ss
from math import *


df = pd.read_csv(r'/Users/theoalegretti/Desktop/_VIZ_MODELS_PRED__202210221314_sample.csv',sep=',')

def log_vraisemblance(mu,sigma,data):
    log = np.log(ss.pdf(data,mu,sigma)).sum()
    return log


def DensiteNormale(x,mu,sigma):
    return 1/(sigma * sqrt(2*pi))*exp(-0.5*((x-mu)/sigma)**2)

def DensiteLAPLACE(x,mu,sigma):
    return (1/2*sigma)*exp((-abs(x-mu)/sigma))


# Test de la stabilité des od et des erreurs 

data = df[((df['ID_OD']== 642) & (df['SECURING_CUM']==1))]

d = data

go.Figure(px.scatter(x=d.ERROR_CC,y=d.ERROR_OCC,trendline='ols')).show()

# ID_TRAIN : 10833     ID_JOUR : 4180

data = df[((df['ID_OD']==642) & (df['ID_TRAIN']==10833)) & (df['ID_JOUR']==4180)]


fig = go.Figure(go.Scatter(x=data['JX'],y=data['PRICE']))
fig.add_trace(go.Scatter(x=data['JX'],y=data['PRICE_PRED']))
fig.show()



fig = px.histogram(df, x="ERROR_OCC",color='ID_OD',title=f'Résidus du modèle du taux de remplissage sur {df.ID_OD.nunique()} OD  - Avec {len(df)} lignes')
fig.show()


fig = px.histogram(df[(df['ID_OD']==642) & (df['ID_EXP']=='Elasticity_model_Prix')], x="ERROR_OCC",color='JX')
fig.show()


fig = px.histogram(df, x="ERROR_OCC")
fig.show()

fig = px.histogram(df, x="ERROR_PRICE",color="ID_OD")
fig.show()

fig = px.histogram(df, x="ERROR_OCC", color="ID_OD")
fig.show()

df_mu = df.groupby('ID_OD')['ERROR_PRICE','ERROR_OCC'].mean()
df_var = df.groupby('ID_OD')['ERROR_PRICE','ERROR_OCC'].var()


def graph_normal_law(data,od,diff_mu,diff_std,out):
    group_labels = ['displot']
    # filtre anti_outlier
    df = data[data.ID_OD == od]
    Q1 =  np.quantile(df.ERROR_PRICE,out[0] )
    Q3 = np.quantile(df.ERROR_PRICE,out[1] )
    val_inf = Q1
    val_supp = Q3
    if val_inf != val_supp :   
        df = df[df['ERROR_PRICE'].between(Q1,Q3)]
    fig2 = ff.create_distplot([np.array(df.ERROR_PRICE)], group_labels, curve_type='normal')
    normal_x = fig2.data[1]['x']
    fig = go.Figure(px.histogram(df, x="ERROR_PRICE", histnorm='probability density'))
    mu = df['ERROR_PRICE'].mean() + diff_mu
    std = df['ERROR_PRICE'].std()+ diff_std
    # std = df['ERROR_PRICE'].std()/2 + diff_std
    ly = [DensiteNormale(x,mu,std) for x in normal_x]
    fig.add_traces(go.Scatter(x=normal_x, y=ly, mode = 'lines',name  = 'normal'))
    print(f'Loi Normal de moyenne {mu} et de variance {std}')
    fig.show()
    
graph_normal_law(df,642,-1,-3,[0,1])

# BOOTSTRAPPING


from scipy.stats import bootstrap

data = (df[(df['ID_OD']==642) & (df['ID_EXP']=='Elasticity_model_Prix')].ERROR_PRICE,)

#calculate 95% bootstrapped confidence interval for median
bootstrap_ci = bootstrap(data, np.mean, confidence_level=0.95,
                         random_state=1, method='percentile')
#view 95% boostrapped confidence interval
print(bootstrap_ci.confidence_interval)

import statsmodels.api as sm 

def create_securing(data,min_diff_cc_ouv):
        data['SECURING'] = 0
        data['SECURING_CUM'] = 0
        def securing(df,min_diff_cc_ouv):
            df['CC_OUV_DIFF'] = df['CC_OUV'].diff()
            df['OCC_PRED'] = df['OCC_LAG_PRED'].shift(-1)
            df['SECURING'][df['CC_OUV_DIFF']< -min_diff_cc_ouv] = 1
            df['SECURING_CUM'] = df['SECURING'].cummax()
            df['OCC_DIFF'] = df['OCC_RATE'].diff()
            if df.SECURING.nunique() == 1 : 
                df['CC_DIFF'] = 0 
            else : 
                df['CC_DIFF'] = df[df['SECURING']==1]['CC_OUV_DIFF'].to_list()[0]
            return df
        data = data.groupby(['ID_JOUR','ID_TRAIN']).apply(lambda df : securing(df,min_diff_cc_ouv))
        return data


data = df[((df['ID_OD'] == 642) & (df['ID_EXP'] == 'PRED_2_03-10_2022'))]

df_secured = create_securing(data,5)

df_secured[df_secured['ERROR_CC_ROUND']==0]['SECURING_CUM'] = 0
go.Figure(px.scatter(x=x,y=y,trendline='ols')).show()




def calculation_elasticity(predictions_df,predictions_df_train,min_diff_cc_ouv):
    test_secured, train_secured = create_securing(predictions_df,min_diff_cc_ouv) ,  create_securing(predictions_df_train,min_diff_cc_ouv)
    df_secured = pd.concat([train_secured[train_secured['SECURING_CUM']==1],test_secured[test_secured['SECURING_CUM']==1]]).dropna(subset=["OCC_PRED"])
    if len(df_secured) != 0 :
        
        df_secured = df_secured[df_secured['CC_DIFF']!= 0]
        
        df_secured[df_secured['ERROR_CC_ROUND']==0]['SECURING_CUM'] = 0
        df_secured= df_secured.dropna(subset=['OCC_DIFF'])
        y = df_secured['OCC_DIFF']
        x = df_secured['SECURING_CUM']
        
        df_secured = df_secured[(df_secured['CC_DIFF'] != 0) & (df_secured['SECURING_CUM']==1)]
        x = df_secured['CC_DIFF'][df_secured['OCC_DIFF']>-0.5].dropna()
        y = df_secured['OCC_DIFF'][df_secured['OCC_DIFF']>-0.5].dropna()    
        try :
            model = sm.OLS(y,x)
            mod = model.fit()
            influence = mod.get_influence()
            cooks = influence.cooks_distance[0]
            seuil = np.quantile(cooks,0.75) - np.quantile(cooks,0.25)
            ind_cooks = pd.DataFrame(cooks)[0].between(np.quantile(cooks,0.25)-(1.5*seuil),np.quantile(cooks,0.75)+(1.5*seuil))
            x , y = x.reset_index(drop=True)[ind_cooks[ind_cooks == True].index] , y.reset_index(drop=True)[ind_cooks[ind_cooks == True].index]
            model = sm.OLS(y,x)
            mod = model.fit()
            df_secured['ELASTICITY_REG'] = mod.params[0]
            df_secured['ELASTICITY_REG_IC_INF'] = mod.conf_int(0.05)[0][0]
            df_secured['ELASTICITY_REG_IC_SUPP'] = mod.conf_int(0.05)[1][0]
            df_secured['P_VAL_REG'] = mod.pvalues[0]
        except :
            df_secured['ELASTICITY_REG'] = 0
            df_secured['ELASTICITY_REG_IC_SUPP'] = 0
            df_secured['ELASTICITY_REG_IC_INF'] = 0
            df_secured['P_VAL_REG'] = 0
        df_secured = df_secured.groupby(['ID_TRAIN','ID_JOUR']).apply(lambda d : elast2(d,3,predictions_df['ABS_ERROR_PRICE'].mean()))
    else :
        df_secured['OCC_DIFF'] = 0
        df_secured['ELASTICITY_REG'] = 0
        df_secured['ELASTICITY_REG_IC_SUPP'] = 0
        df_secured['ELASTICITY_REG_IC_INF'] = 0
        df_secured['P_VAL_REG'] = 0
        df_secured['ELASTICITY_TDOJ'] = 0
    df_secured['Month']  = pd.DatetimeIndex(df_secured['DATEJ']).month
    df_secured['Week'] = pd.DatetimeIndex(df_secured['DATEJ']).week
    return df_secured, test_secured , train_secured













import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm




y = np.array(df_mu.ERROR_PRICE)
sigma = np.array(df_var.ERROR_PRICE)
J = len(y)

with pm.Model() as pooled:
    # Latent pooled effect size
    mu = pm.Normal("mu", 0, sigma=1e6)
    obs = pm.Normal("obs", mu, sigma=sigma, observed=y)
    trace_p = pm.sample(2000)

az.plot_trace(trace_p)

with pm.Model() as hierarchical:
    eta = pm.Normal("eta", 0, 1, shape=J)
    # Hierarchical mean and SD
    mu = pm.Normal("mu", 0, sigma=10)
    tau = pm.HalfNormal("tau", 10)
    # Non-centered parameterization of random effect
    theta = pm.Deterministic("theta", mu + tau * eta)
    obs = pm.Normal("obs", theta, sigma=sigma, observed=y)
    trace_h = pm.sample(2000, target_accept=0.9)

az.plot_trace(trace_h, var_names="mu")
# az.plot_forest(trace_h, var_names="theta")
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
####################################################################################################################################### 

# PSO new - ZONE 51 

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
            particules[p,0] = params['vit']*speed[p,0] + params['c1']*random.random()*(particule_best[0]-particules[p,ite]) + params['c2']*random.random()*(particules[p,ite-1]-particules[p,ite])
            particules[p,1] =  params['vit']*speed[p,0] + params['c1']*random.random()*(particule_best[1]-inx[r2,nbv]) + params['c2']*random.random()*(inx[locc,nbv]-inx[r2,nbv])
            value_fct[p,0] = fct(data,particules[p,0],particules[p,1])

