# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator,img_to_array,array_to_img,load_img

import h5py
import os

import tensorflow as tf
tf.python.control_flow_ops = tf

import numpy as np

#import cv2, numpy as np

def load_weights(weights_path,model):  
    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    #不要最后的FC层


    model.add(Flatten())
    
    if weights_path:
        load_weights(weights_path,model)

    return model


    
model = VGG_16("/home/max/下载/vgg16_weights.h5")

#不用rescale更好！！
#datagen = ImageDataGenerator(rescale=1./255)
datagen = ImageDataGenerator()

#注意这里的batch_size=32,跟下面的top fit对应，否则无法训练top
train_data = datagen.flow_from_directory("/home/max/data/train",target_size=(224,224)\
                                         ,batch_size=32,shuffle=False,class_mode=None)

test_data = datagen.flow_from_directory("/home/max/data/test",target_size=(224,224)\
                                         ,batch_size=32,shuffle=False,class_mode=None)

train_out = model.predict_generator(train_data,val_samples=2000)

test_out = model.predict_generator(test_data,val_samples=800)


np.save(open("/home/max/train.out",'w'),train_out)
np.save(open("/home/max/test.out",'w'),test_out)

def build_top():
    train_feature = np.load("/home/max/train.out")
    test_feature = np.load("/home/max/test.out")
    
    train_label = np.array([0]*1000+[1]*1000)
    test_label = np.array([0]*400+[1]*400)
    
    model = Sequential()
    #model.add(Flatten(input_shape = (1000,)))
    model.add(Dense(256,activation='relu',input_shape=train_out.shape[1:]))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='sigmoid'))
    
    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
    
    model.fit(train_feature,train_label,validation_data=(test_feature,test_label),nb_epoch=50)
    
    return model
    
    
top_model = build_top()

#到此，实验2完成，top训练达到90%


model.add(top_model)

#冻结前25层
for layer in model.layers[:25]:
    layer.trainable = False
    
model.compile(optimizer=SGD(lr=1e-4,momentum=0.9),loss='binary_crossentropy',metrics=['accuracy'])


#重新初始化训练集生成器
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        "/home/max/data/train/",
        target_size=(224,224),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        "/home/max/data/test/",
        target_size=(224,224),
        batch_size=32,
        class_mode='binary')
#至此，完成实验3，精度达到95%
model.fit_generator(
        train_generator,
        samples_per_epoch=2000,
        nb_epoch=50,
        validation_data=validation_generator,
        nb_val_samples=800)
