# coding=utf-8

import os
#import tensorflow as tf
from keras.applications import VGG16
from keras.preprocessing import image
#import matplotlib.pyplot as plt
import numpy as np 
from keras.applications.imagenet_utils import *
from keras import models
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import optimizers
import sys
from keras.models import model_from_json

image_size = 64

def uuVGG16_eva(model_path,test_dir, test_batchsize=32):
        scores = None
        try:
                #load trained model from disk
                model = load_model(model_path)
                #model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

                #load images of test
                try:
                        #create test generator for test images
                        test_datagen = ImageDataGenerator(rescale=1./255)

                        #loop through test folder
                        test_generator  = test_datagen.flow_from_directory(
                                test_dir,
                                target_size=(image_size, image_size),
                                batch_size=test_batchsize,
                                class_mode='binary')

                        #Do evaluation via evaluate generator
                        '''FROM Keras Document 

                                evaluate_generator(generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)

                                Evaluates the model on a data generator.
                                The generator should return the same kind of data as accepted by test_on_batch.

                                Arguments

                                generator: Generator yielding tuples (inputs, targets) or (inputs, targets, sample_weights) or an instance of Sequence (keras.utils.Sequence) object in order to avoid duplicate data when using multiprocessing.
                                steps: Total number of steps (batches of samples) to yield from generator before stopping. Optional for Sequence: if unspecified, will use the len(generator) as a number of steps.
                                max_queue_size: maximum size for the generator queue
                                workers: Integer. Maximum number of processes to spin up when using process based threading. If unspecified, workers will default to 1. If 0, will execute the generator on the main thread.
                                use_multiprocessing: if True, use process based threading. Note that because this implementation relies on multiprocessing, you should not pass non picklable arguments to the generator as they can't be passed easily to children processes.
                                verbose: verbosity mode, 0 or 1.
                        '''
                        scores = model.evaluate_generator(test_generator,verbose = 0, steps = len(test_generator))
                        i=0
                        for s in scores:
                            print("{}: {:.6f}".format(model.metrics_names[i], s))
                            i=i+1

                except IOError:
                        print("Error: image path not found!")

        except IOError:
                print("Error: model path not found!")
                
        #only output accuracy
        return scores[1]

def main():
        model_path = r"D:\00_Tasks\05_G project\PCB_AI_Inspection\PCB_AI_Inspection\Support Files\pythonwork\Models\glass_flaw_vgg\glass_flaw_vgg.h5"
        test_dir = r"D:\00_Tasks\05_G project\PCB_AI_Inspection\PCB_AI_Inspection\Support Files\pythonwork\Models\glass_flaw_vgg\Data\Train"
        print (uuVGG16_eva(model_path, test_dir,100))

if __name__=='__main__':
  main()
