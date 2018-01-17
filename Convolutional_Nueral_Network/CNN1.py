import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

#Weights

def init_weights(shape):
	init_func= tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(init_func)

#Bias

def init_bias(shape):
	init_func1= tf.constant(0.1,stddev=0.1)
	return tf.Variable(init_func)

# 2D Convolution

def conv_2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2by2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')

def convolutional_layer(input_x, shape):
	W= init_weights(shape)
	b= init_bias([shape[3]])
	return tf.nn.relu(conv2d(input_x,W)+b)

def normal_full_layer(input_layer, size):
	input_size= int(input_layer.getsize()[1])
	W= init_weights([input_size,size])
	b= init_bias([size])
	return tf.matmul(input_layer,W)+b

#Placeholders

x_data= tf.placeholder(tf.float32, shape=[None,784])
y_true= tf.placeholder(tf.float32,shape=[None,10])

#Layers

x_image= tf.reshape(x_data[-1,28,28,1])

convol_layer1= convolutional_layer(x_image,shape=[6,6,32,1])
convol_layer1pool= max_pool_2by2(convol_layer1)

convol_layer2= convolutional_layer(convol_layer1pool,shape=[6,6,64,32])
convol_layer2pool2= max_pool_2by2(convol_layer2)

convo_2_flat = tf.reshape(convol_layer2pool2,[-1,7*7*64])
full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat,1024))

#Hold prob

hold_prob= tf.placeholder(tf.float32)
full_dropout= tf.nn.dropout(full_layer_on,keep_prob=hold_prob)

y_pred= normal_full_layer(full_dropout,10)

#LOSS
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))

#Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
train = optimizer.minimize(cross_entropy)

#Initilaize Variables
init = tf.global_variables_initializer()

#Session
steps = 5000

with tf.Session() as sess:
    
    sess.run(init)
    
    for i in range(steps):
        
        batch_x , batch_y = mnist.train.next_batch(50)
        
        sess.run(train,feed_dict={x:batch_x,y_true:batch_y,hold_prob:0.5})
        
        # PRINT OUT A MESSAGE EVERY 100 STEPS
        if i%100 == 0:
            
            print('Currently on step {}'.format(i))
            print('Accuracy is:')
            # Test the Train Model
            matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))

            acc = tf.reduce_mean(tf.cast(matches,tf.float32))

            print(sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels,hold_prob:1.0}))
            print('\n')




