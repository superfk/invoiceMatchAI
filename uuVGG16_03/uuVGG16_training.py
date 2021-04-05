#!/usr/bin/env python
# -*- coding: utf-8 -*-

import keras
from keras.applications import VGG16
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
import sys
import csv
from time import time

#bDataAugumentUsed = False
g_history = None

#resize image
image_size = 64

#create preview folder if not exists
def create_folder(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)
  return directory

#customize callback
class TrainHistory(keras.callbacks.Callback):
  def __init__ (self,csvpath):
    self.csvpath = csvpath
  def on_train_begin(self, logs={}):
    self.losses = []
  def on_epoch_end(self, epoch, logs={}):
    self.losses.append(logs.get('loss'))
    with open(self.csvpath, 'w', newline='') as csvfile:
      fieldnames = logs.keys()
      writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
      writer.writeheader()
      writer.writerow(logs)


def uuGetTrainGoodPath(input_path):
  return os.path.join(input_path, r'Data\Train\Good')
  
def uuGetTrainBadPath(input_path):
  return os.path.join(input_path, r'Data\Train\Bad')

def uuGetValidationGoodPath(input_path):
  return os.path.join(input_path, r'Data\Validation\Good')
  
def uuGetValidationBadPath(input_path):
  return os.path.join(input_path, r'Data\Validation\Bad')

def uuGetTestGoodPath(input_path):
  return os.path.join(input_path, r'Data\Test\Good')
  
def uuGetTestBadPath(input_path):
  return os.path.join(input_path, r'Data\Test\Bad')

def extract_features(directory, sample_count, datagen, batch_size, conv_base):
    # shape=(sample_count,7,7,512) is because the last layer shape of Vgg16 is (7,7,512)
    #print(final_layer_shape_size = conv_base.get_layer("block5_pool").output.shape[0])
    x=conv_base.get_layer("block5_pool").output.shape[1].value
    y=conv_base.get_layer("block5_pool").output.shape[2].value
    c=conv_base.get_layer("block5_pool").output.shape[3].value
    features = np.zeros(shape=(sample_count, x, y, c))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='binary',
        save_to_dir=create_folder(r'D:/LogData/PCBImage/train normal preview'),
        save_format='jpeg')
    i = 0
    for inputs_batch, labels_batch in generator:
        #print(inputs_batch)
        features_batch = conv_base.predict(inputs_batch)
        #print(features_batch.shape)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        print(directory)
        print(labels_batch)
        print(len(labels_batch))
        print(labels_batch.shape)
        print("\n")
        i += 1
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    features = np.reshape(features,(sample_count, x*y*c))
    return features, labels





def uuVGG16_training(model_path, input_path, epoch = 30, batch_size = 32, 
  bDataAugumentUsed = None, train_batchsize = None, val_batchsize = None):
  global g_history
  start_time = time()

  #base_dir = 'Data'
  base_dir = os.path.join(input_path, r'Data')

  train_dir = os.path.join(base_dir, 'Train')
  validation_dir = os.path.join(base_dir, 'Validation')
  test_dir = os.path.join(base_dir, 'Test')

  #sTrainGoodPath = os.path.join(input_path, r'\Data\Train\Good')
  sTrainGoodPath = os.path.join(input_path, r'Data\Train\Good')
  print('sTrainGoodPath={}'.format(sTrainGoodPath))
  pathTrainGood, dirsTrainGood, filesTrainGood = next(os.walk(sTrainGoodPath))    #r'C:\pythonwork\uuVGG16_03\Data\Train\Good'
  file_count_train_good = len(filesTrainGood)

  sTrainBadPath = os.path.join(input_path, r'Data\Train\Bad')
  pathTrainBad, dirsTrainBad, filesTrainBad = next(os.walk(sTrainBadPath))         #r'C:\pythonwork\uuVGG16_03\Data\Train\Bad'
  file_count_train_bad = len(filesTrainBad)

  train_sum = file_count_train_good + file_count_train_bad

  sTestGoodPath = os.path.join(input_path, r'Data\Test\Good')
  pathTestGood, dirsTestGood, filesTestGood = next(os.walk(sTestGoodPath))    #r'C:\pythonwork\uuVGG16_03\Data\Test\Good'
  file_count_test_good = len(filesTestGood)

  sTestBadPath = os.path.join(input_path, r'Data\Test\Bad')
  pathTestBad, dirsTestBad, filesTestBad = next(os.walk(sTestBadPath))    #r'C:\pythonwork\uuVGG16_03\Data\Test\Bad'
  file_count_test_bad = len(filesTestBad)

  test_sum = file_count_test_good + file_count_test_bad

  svalidationGoodPath = os.path.join(input_path, r'Data\Validation\Good')
  pathvalidationGood, dirsvalidationGood, filesvalidationGood = next(os.walk(svalidationGoodPath))    #r'C:\pythonwork\uuVGG16_03\Data\validation\Good'
  file_count_validation_good = len(filesvalidationGood)

  svalidationBadPath = os.path.join(input_path, r'Data\Validation\Bad')
  pathvalidationBad, dirsvalidationBad, filesvalidationBad = next(os.walk(svalidationBadPath))    #r'C:\pythonwork\uuVGG16_03\Data\validation\Bad'
  file_count_validation_bad = len(filesvalidationBad)

  validation_sum = file_count_validation_good + file_count_validation_bad

  '''
  keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

  - input_shape: optional shape tuple, only to be specified if include_top is False 
    (otherwise the input shape has to be (224, 224, 3) (with 'channels_last' data format) or (3, 224, 224) 
    (with 'channels_first' data format). It should have exactly 3 inputs channels, and width and height should be no smaller than 32. E.g. (200, 200, 3) 
    would be one valid value.

  - include_top: whether to include the 3 fully-connected layers at the top of the network.

  - pooling: Optional pooling mode for feature extraction when include_top is False.
    None means that the output of the model will be the 4D tensor output of the last convolutional layer.
    'avg' means that global average pooling will be applied to the output of the last convolutional layer, and thus the output of the model will be a 2D tensor.
    'max' means that global max pooling will be applied.
  '''
  vgg_conv = VGG16(weights='imagenet', include_top=False, 
    input_shape=(image_size, image_size, 3))
  # Freeze the layers except the last 4 layers
  for layer in vgg_conv.layers[:-4]:
      layer.trainable = False
      #print(layer.get_config())
  print("Vgg16 model summary:")
  vgg_conv.summary()

  #Prepare Trainiing data
  datagen = ImageDataGenerator(rescale=1./255)
  train_features, train_labels = extract_features(train_dir, train_sum, datagen, batch_size, vgg_conv)  #90+90=180   #70+95=165
  validation_features, validation_labels = extract_features(validation_dir, validation_sum, datagen, batch_size, vgg_conv)  #10+10=20    #12+12
  test_features, test_labels = extract_features(test_dir, test_sum, datagen, batch_size, vgg_conv)  #2+6=8     #20+20=40

  #Create new Model
  model = models.Sequential()

  #Add the vgg convolutional base model
  model.add(vgg_conv)

  #Get shape for VGG16
  x = vgg_conv.get_layer("block5_pool").output.shape[1].value
  print(type(x))
  y = vgg_conv.get_layer("block5_pool").output.shape[2].value
  c = vgg_conv.get_layer("block5_pool").output.shape[3].value

  # Add new layers
  model.add(layers.Flatten(input_shape=vgg_conv.output_shape[1:]))
  model.add(layers.Dense(256, activation='relu', input_dim = x*y*c))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(1, activation='sigmoid'))

  #Prepare original Training data
  train_original_datagen = ImageDataGenerator(rescale=1./255)

  #Prepare Augment Training data
  train_aug_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      zoom_range=[-0.2,0.2],
      horizontal_flip=True,
      fill_mode='nearest')

  validation_original_datagen = ImageDataGenerator(rescale=1./255)

  #Prepare Augment validation data
  validation_aug_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      zoom_range=[-0.2,0.2],
      horizontal_flip=True,
      fill_mode='nearest')

  #test data keeps the same
  test_datagen = ImageDataGenerator(rescale=1./255)

  #define generator flow_from_directory method for original images
  train_original_generator = train_original_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='binary',
        #save_to_dir=create_folder(r'D:/LogData/PCBImage/train original preview'),
        save_format='jpeg'
        )
  #define generator flow_from_directory method for data augmentation images
  train_aug_generator = train_aug_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=train_batchsize,
        class_mode='binary',
        #save_to_dir=create_folder(r'D:/LogData/PCBImage/train augmentation preview'),
        save_format='jpeg'
        )

  #define generator flow_from_directory method for original images
  validation_original_generator = validation_original_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='binary',
        #save_to_dir=create_folder(r'D:/LogData/PCBImage/valid original preview'),
        save_format='jpeg',
        shuffle=False)

  #define generator flow_from_directory method for augment images
  validation_aug_generator = validation_aug_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=val_batchsize,
        class_mode='binary',
        #save_to_dir=create_folder(r'D:/LogData/PCBImage/valid augmentation preview'),
        save_format='jpeg',
        shuffle=False) 

  test_generator  = test_datagen.flow_from_directory(
        test_dir,
        target_size=(image_size, image_size),
        batch_size=val_batchsize,
        class_mode='binary',
        #save_to_dir=create_folder(r'D:/LogData/PCBImage/test original preview'),
        save_format='jpeg',
        shuffle=False)

  #Compile the model
  model.compile(optimizer=optimizers.RMSprop(lr=0.00001), loss='binary_crossentropy', metrics=['acc'])
  model.summary()

  #Setup callback function
  #CSV callback
  csv_logger = CSVLogger(filename=model_path+'_training.log')
  #Customize callback
  custLogger = TrainHistory(model_path+'_output.csv')
  #Tensorborad callback
  log_dir = os.path.split(model_path)[0]
  tensorboard = TensorBoard(log_dir=log_dir+"/logs/{}".format(time()))
  #Early stopping callback
  early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=0, mode='auto' )

  #Train the model
  if(True == bDataAugumentUsed):
      g_history = model.fit_generator(train_aug_generator,
      steps_per_epoch=train_aug_generator.samples/train_aug_generator.batch_size, 
      epochs=epoch, 
      validation_data=validation_aug_generator, 
      validation_steps=validation_aug_generator.samples/validation_aug_generator.batch_size, 
      verbose=2,
      callbacks=[tensorboard,csv_logger,early_stop])
  else:
      g_history = model.fit_generator(train_original_generator,
      steps_per_epoch=train_original_generator.samples/train_original_generator.batch_size, 
      epochs=epoch, 
      validation_data=validation_original_generator, 
      validation_steps=validation_original_generator.samples/validation_original_generator.batch_size, 
      verbose=2,
      callbacks=[tensorboard,csv_logger,early_stop])

  #End Training and calculate training time in second
  end_time=time()
  train_time = end_time - start_time

  # Save Model
  json_string = model.to_json()
  szModelPath = os.path.splitext(model_path)[0]
  json_pathfilename = szModelPath + '.json'
  open(json_pathfilename, 'w').write(json_string)              #'Models/uuvgg16_keras.json'
  #model.save_weights(model_path)                               #'Models/uuvgg16_keras.hdf'
  model.save(szModelPath + '.h5')
  uuShowChart()
  return train_time


def uuShowChart():
  import matplotlib.pyplot as plt
  global g_history
  acc = g_history.history['acc']
  val_acc = g_history.history['val_acc']
  loss = g_history.history['loss']
  val_loss = g_history.history['val_loss']

  epochs = range(len(acc))

  plt.plot(epochs, acc, 'bo', label='Training acc')
  plt.plot(epochs, val_acc, 'b', label='Validation acc')
  plt.title('Training and validation accuracy')
  plt.legend()

  plt.figure()

  plt.plot(epochs, loss, 'bo', label='Training loss')
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title('Training and validation loss')
  plt.legend()

  plt.show()
 

#######################################################################################################
#uuNote:below is for testing by batch file used.  Debug purpose!!!
def uuShowImage(img):  
  plt.axis('off')                                         #圖像座標軸不顯示
  plt.imshow(img)                                         #將圖像載入plt內(尚未顯示)
  plt.show(img)                                           #秀出圖像 
  
#開始對sys.argv下定義
def main():
    model_path = r"D:\00_Tasks\05_G project\PCB_AI_Inspection\PCB_AI_Inspection\Support Files\pythonwork\uuVGG16_03\Models\debug\debug_vgg16_3.h5"
    input_path = r"D:\00_Tasks\05_G project\PCB_AI_Inspection\PCB_AI_Inspection\Support Files\pythonwork\Models\capalost_zoom15x_vgg16"
    print (uuVGG16_training(model_path, input_path, epoch = 5, batch_size = 32,bDataAugumentUsed = True, train_batchsize = 30, val_batchsize = 10))

if __name__=='__main__':
    main()
