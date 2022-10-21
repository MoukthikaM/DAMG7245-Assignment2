from fastapi import FastAPI
#needed packages 
import xarray as xr
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
# import jsonify
#load model from sklearn
import json
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

#plot parameters that I personally like, feel free to make these your own.
import matplotlib
matplotlib.rcParams['axes.facecolor'] = [0.9,0.9,0.9] #makes a grey background to the axis face
matplotlib.rcParams['axes.labelsize'] = 14 #fontsize in pts
matplotlib.rcParams['axes.titlesize'] = 14 
matplotlib.rcParams['xtick.labelsize'] = 12 
matplotlib.rcParams['ytick.labelsize'] = 12 
matplotlib.rcParams['legend.fontsize'] = 12 
matplotlib.rcParams['legend.facecolor'] = 'w' 
matplotlib.rcParams['savefig.transparent'] = False

#make default resolution of figures much higher (i.e., Higxh definition)
# %config InlineBackend.figure_format = 'retina'

#import some helper functions for our other directory.

app = FastAPI()

@app.get("/health")
def root():
    return {"message": "Hello World"}




# @app.get("/model")
# async def regression():
import sys

sys.path.insert(1, '../scripts/')

from aux_functions import load_n_combine_df
(X_train,y_train),(X_validate,y_validate),(X_test,y_test) = load_n_combine_df(path_to_data='../datasets/sevir/',features_to_keep=np.arange(0,36,1),class_labels=False,dropzeros=True)

  #create scaling object 
scaler = StandardScaler()
  #fit scaler to training data
scaler.fit(X_train)

  #transform feature data into scaled space 
X_train = scaler.transform(X_train)
X_validate = scaler.transform(X_validate)
X_test = scaler.transform(X_test)

  #double check that mean is 0 and std is 1. 
np.mean(X_train,axis=0),np.std(X_train,axis=0)

#initialize
model = LinearRegression()
  
print(model)
import pickle
model = model.fit(X_train,y_train)
pickle.dump(model, open('model.pkl','wb'))
#get predictions 
# X_test = [0.0346,0.0473,0.0609,0.0739,0.1228,0.362,0.7824,0.8911,1.0898]
# X_test=X_test+X_test+X_test+X_test
# X_test=np.asarray(X_test)
# B = np.reshape(X_test, (1, -1))

from gewitter_functions import get_mae,get_rmse,get_bias,get_r2


def predictflash(xtest):
   
   yhat = model.predict(xtest)
   print(yhat)
  #  mae = get_mae(y_validate,yhat)
  #  rmse = get_rmse(y_validate,yhat)
  #  bias = get_bias(y_validate,yhat)
  #  r2 = get_r2(y_validate,yhat)
  #  json.dumps(yhat.tolist())
  #  #print them out so we can see them 
  #  print('MAE:{} flashes, RMSE:{} flashes, Bias:{} flashes, Rsquared:{}'.format(np.round(mae,2),np.round(rmse,2),np.round(bias,2),np.round(r2,2)))
  #  # return json.dumps(yhat.tolist())

   return yhat







