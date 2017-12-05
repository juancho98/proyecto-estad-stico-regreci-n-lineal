#RSE.py

import pandas as pd
advert = pd.read_csv("./dataBases/proy.csv")

from sklearn.feature_selection import RFE
from sklearn.svm import SVR #importar SVR
feature_cols = ['alim17_1','alim17_2','alim17_3', 'alim17_4', 'alim17_5','alim17_6','alim17_7','alim17_8','alim17_9','alim17_10','alim17_11', 'num_auto', 'num_moto', 'num_bici',  'num_radio', 'num_tvd', 'num_licua', 'num_micro', 'num_refri', 'num_estuf', 'num_lavad', 'num_planc', 'num_venti', 'num_compu', 'num_impre', 'num_juego']
#
X = advert[feature_cols]
Y = advert['ing_tri']
estimator = SVR(kernel="linear")

selector = RFE(estimator, 2, step=1) #2 es para los niveles 
selector = selector.fit(X,Y)  #llenarlo con x y y
#regresion lineal 
A = selector.support_
B = selector.ranking_
#lista de variables seleccionadas
print(selector.support_) # el _ es el nombre del metodo
#el mejor modelo sera ->  crear las matrices de correlacion, quitar las variables con coeficiente de correlacion mas bajo,
print(selector.ranking_)
