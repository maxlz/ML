# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 19:50:54 2016

@author: max
"""

import tensorflow as tf
import numpy as np

import matplotlib.pylab as m

x_data = np.linspace(0.0,1.0,num = 500,dtype='float32')
x_data = np.reshape(x_data,(500,))
y_data = np.linspace(0.0,1.0,num = 500,dtype='float32')
y_data = y_data + np.random.rand(500,)/10+1

x = tf.placeholder(dtype='float32')
y = tf.placeholder(dtype='float32')

W = tf.Variable(tf.random_uniform((1,1),-1,1),name='W',dtype='float32')
b = tf.Variable(tf.random_uniform((1,1),-1,1),name='b',dtype='float32')

Y = W*x+b

loss = tf.reduce_mean(tf.square(Y-y))

opt = tf.train.RMSPropOptimizer(0.01).minimize(loss)

init = tf.initialize_all_variables()

ses = tf.Session()

ses.run(init)

for i in range(1000):
    ses.run(opt,feed_dict={x:x_data,y:y_data})
    if i%50 == True:
        print loss.eval(session = ses,feed_dict={x:x_data,y:y_data})
        B = b.eval(session = ses)
        final_w = W.eval(session = ses)
        print B,final_w
        
final_y = np.multiply(x_data,final_w) + B
final_y = np.reshape(final_y,(500,))
m.plot(x_data,final_y)
m.plot(x_data,y_data,'r+')
    
