# import os
# import pandas as pd
# import h5py # needs conda/pip install h5py
# import matplotlib.pyplot as plt
# from curses import flash
# from urllib import response
import numpy as np
# from jupyter_notebooks.app import predict
# import xarray as xr
# from numpy import asarray
from sklearn import preprocessing
import numpy as np
# import  app as app
from fastapi import FastAPI
from PIL import Image
# image = Image.open('../vil.png')
import pickle
import pytest
app = FastAPI()

from pydantic import BaseModel

class Images(BaseModel):
    vis: str
    vil: str
    ir069: str
    ir107: str


def imagepreprocessing(path):
    # path='../'+events+'.png'
    print(path)
    im_gray = np.array(Image.open(path),np.float64)
    percent=percentile(preprocessing.normalize(im_gray*1e-4))
    print((im_gray))
    print(im_gray.shape)
    return percent


def percentile(data_sub):
   desired_percentiles = np.array([0,1,10,25,50,75,90,99,100])
   percentiles = np.nanpercentile(data_sub,desired_percentiles,axis=(0,1))
   percentiles = np.reshape(percentiles, (1, -1))
   return percentiles




from fastapi import Response
import json


@app.post("/flashes")
async def flashes(images: Images):
    vis = imagepreprocessing(images.vis)
    ir069 = imagepreprocessing(images.ir069)
    ir107 = imagepreprocessing(images.ir107)
    vil = imagepreprocessing(images.vil)
    X_test=np.concatenate((ir107,ir069,vis,vil),axis=1)
    print(X_test.shape)
    flashes = predictflash((X_test))
    return {'flashes': flashes}




def predictflash(X_test):
      model = pickle.load(open('model.pkl','rb'))
      print(model)
      flashes=model.predict(X_test)
      print(flashes[0])
      return flashes[0]


@app.post("/images")
async def create_item(images: Images):
    vis = imagepreprocessing(images.vis)
    return images



@app.get("/")
async def get():
     return {"msg": "Hello World"}