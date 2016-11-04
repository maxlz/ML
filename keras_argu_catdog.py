# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 22:38:04 2016

@author: max
"""

import tensorflow as tf

tf.python.control_flow_ops = tf

import keras
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

'''
datagen = ImageDataGenerator(rotation_range=40,width_shift_range=0.2
,height_shift_range=0.2,shear_range=0.2,zoom_range=0.1,rescale=1./225
,horizontal_flip=True)

img = load_img('./data/train/dogs/dog.1.jpg')
x = img_to_array(img)
x = x.reshape((1,)+x.shape)


i=0
for batch in datagen.flow(x,batch_size=1,save_to_dir='./data/argumented',save_prefix='arg_'):
    i+=1
    if i>4:
        break

'''

'''
for i in range(5):
    print i
    datagen.flow(x,batch_size=1,save_to_dir='./data/argumented',save_prefix='arg_')
'''

model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(3, 150, 150)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop'
              , metrics=['accuracy'])

data_train_gen = ImageDataGenerator(rotation_range=40
                                    , width_shift_range=0.2, height_shift_range=0.2, rescale=1. / 255
                                    , shear_range=0.3, horizontal_flip=True)

data_test_gen = ImageDataGenerator(rotation_range=0.3, rescale=1. / 255)

train_flow = data_train_gen.flow_from_directory('/home/max/data/train/', target_size=(150, 150)
                                                , batch_size=32, class_mode='binary')

test_flow = data_test_gen.flow_from_directory('/home/max/data/test/', target_size=(150, 150)
                                              , batch_size=32, class_mode='binary')

# samples_per_epoch最好是全部数据样例的总和，如2000张猫狗，则设置为2000
model.fit_generator(train_flow, samples_per_epoch=3200, nb_epoch=50
                    , validation_data=test_flow, nb_val_samples=800)

model.save_weights('keras_arg_dagcat.h5')
