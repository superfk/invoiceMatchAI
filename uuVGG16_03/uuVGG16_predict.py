# coding=utf-8

import os
#import tensorflow as tf
from keras.applications import VGG16
from keras.preprocessing import image
#import matplotlib.pyplot as plt
import numpy as np 
from keras.applications.imagenet_utils import *
from keras import models
from keras import layers
from keras import optimizers
import sys
from keras.models import model_from_json
from time import time

#assign g_model as global variable to memorize loaded models
g_model_dict = dict()
g_model = None

image_size = 64

def loadModel(model_path):
	#g_model_dict is global variable
	global g_model_dict

	#get model filename without .h5
	model_name = os.path.splitext(model_path)[0]

	#check if model already in model dictionary, if yes, return stored model object, else add into dict
	if model_name in g_model_dict:
		currentModel = g_model_dict[model_name]
	else:
		g_model_dict[model_name] = models.load_model(model_name + '.h5')
		currentModel = g_model_dict[model_name]

	#final return loaded model
	return currentModel

def uuPredict(model_path,img_path):
  img = None
  sRet = None
  dOutcome = 0.0
  global g_model
  g_szStatus = None
  
  #Load image from file
  try:
    img = image.load_img(img_path, target_size=(image_size, image_size))
  except IOError:
    g_szStatus = '1;uuNote:status:{0}::-->img_path is NOT Exist!!!'.format(img_path)
    print(g_szStatus)
    return g_szStatus

  # Extract features with VGG16 (https://keras.io/applications/#extract-features-with-vgg16)
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x, mode="tf")
  
  # Load Model
  try:
    #szModelPath = os.path.splitext(model_path)[0]
    start_time = time()
    #g_model = models.load_model(szModelPath + '.h5')
    g_model = loadModel(model_path)
    print("Loading model time(s):{:.3f}".format(time()-start_time))
  except IOError:
    g_szStatus = '2;uuNote:status:{0}::-->model_path is NOT Exist!!!'.format(model_path)
    print(g_szStatus)
    return g_szStatus

  # Prediction
  preds = g_model.predict(x)
  dOutcome = preds[0][0]
  g_szStatus = '0;{0}'.format(dOutcome)          #'{0};uuNote:status:predict OK'.format(dOutcome)
  return g_szStatus
  
def main():
  model_path = r"D:\00_Tasks\05_G project\PCB_AI_Inspection\PCB_AI_Inspection\Support Files\pythonwork\Models\capalost_15X_B1DVT_vgg\capalost_15X_B1DVT_vgg.h5"
  img_path = r"D:\00_Tasks\05_G project\PCB_AI_Inspection\PCB_AI_Inspection\Support Files\pythonwork\Models\capalost_15X_B1DVT_vgg\Data\Test\Bad\0402_84.jpg"
  print (uuPredict(model_path, img_path))

if __name__=='__main__':
  main()
