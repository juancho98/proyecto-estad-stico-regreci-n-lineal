# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 16:16:00 2017

@author: karen
"""


import pandas as pd
import statsmodels.formula.api as smf 
advert = pd.read_csv("./dataBases/proy.csv") 
model2=smf.ols(formula='ing_tri~alim17_4+alim17_6', #marco de datos pandas
               data=advert).fit() #obtenemos el modelo, formula y marco de datos 
               #fit -> ajustar datos al modelos, ols minimos cuadrados 
print(model2.params)  #alpha y betas 
print(model2.pvalues) #los p valores
print(model2.rsquared) #porcentaje del error total que describe 

sales_pred=model2.predict(advert[['alim17_4','alim17_6']]) #marco de datos advertising
print(sales_pred.head())

a = model2.params[0]
b_tv = model2.params[1]
b_radio = model2.params[2]
#RSE
import numpy as np
advert['sales_pred'] = a + b_tv * advert['alim17_4'] + b_radio * advert['alim17_6'] #cuando imprimes model2
                         #\ para saltos de linea 
advert['SSD'] = (advert['ing_tri'] - \
                  advert['sales_pred'])**2
SSD=advert.sum()['SSD']
n = len(advert["ing_tri"])
print("n",n)
p = 2 #numero de categorias, 2 = tv && newspaper
RSE=np.sqrt(SSD/(n-p-1))
print("RSE", RSE)
salesmean=np.mean(advert['ing_tri'])
print("salesmean", salesmean)
error=RSE/salesmean
print("error", error)
print(4*"\n")
print(model2.summary())#resumen de estadisticos 