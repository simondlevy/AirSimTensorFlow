#!/usr/bin/env python3
'''
collision_training.py : uses stored images to training a neural net to detect collisions

Copyright (C) 2017 Jack Baird, Alex Cantrell, Keith Denning, Rajwol Joshi, 
Simon D. Levy, Will McMurtry, Jacob Rosen

This file is part of AirSimTensorFlow

MIT License
'''

# Built-in modules
import tensorflow as tf
import numpy as np
import pickle

# Modules for this project
from image_helper import loadgray
from tf_softmax_layer import inference

# Final image is crash; previous are no-crash
SAFESIZE = 5

# Where we've stored images
IMAGEDIR = './carpix'

# Where we'll store weights and biases
PARAMFILE = 'params.pkl'

# Parameters
learning_rate = 0.01
training_epochs = 500
batch_size = 100
display_step = 10

def loss(output, y):
    dot_product = y * tf.log(output)

    # Reduction along axis 0 collapses each column into a single
    # value, whereas reduction along axis 1 collapses each row 
    # into a single value. In general, reduction along axis i 
    # collapses the ith dimension of a tensor to size 1.
    xentropy = -tf.reduce_sum(dot_product, axis=1)
     
    loss = tf.reduce_mean(xentropy)

    return loss

def training(cost, global_step):

    tf.summary.scalar('cost', cost)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(cost, global_step=global_step)

    return train_op

def main():

    # This will get number of pixels in each image (they must all be the same!)
    imgsize = 0

    # Read in images from car, convert to grayscale, scale down, and flatten for use as input
    images = []
    for k in range(SAFESIZE):

        image = loadgray(IMAGEDIR + '/image%03d.png' % k)
                
        imgsize = np.prod(image.shape)
        images.append(image)

    # All but last image is safe (01 = no-crash; 10 = crash)
    targets = []
    for k in range(SAFESIZE-1):
        targets.append([0,1])
    targets.append([1,0])
    
    with tf.Graph().as_default():

        x = tf.placeholder('float', [None, imgsize]) # car FPV images
        y = tf.placeholder('float', [None, 2])       # 01 = no-crash; 10 = crash

        output = inference(x, imgsize, 2)

        cost = loss(output, y)

        global_step = tf.Variable(0, name='global_step', trainable=False)

        train_op = training(cost, global_step)

        sess = tf.Session()

        init_op = tf.global_variables_initializer()

        sess.run(init_op)

        # Training cycle
        for epoch in range(training_epochs):
                
            # Fit training using batch data
            sess.run(train_op, feed_dict={x: images, y: targets})
                
            # Compute average loss
            avg_cost = sess.run(cost, feed_dict={x: images, y: targets})
                
            # Display logs per epoch step
            if epoch%display_step == 0:
                print('Epoch:', '%04d' % epoch, 'cost =', '{:.9f}'.format(avg_cost))

        print('Optimization Finished; saving weights to ' + PARAMFILE)
        params = [sess.run(param) for param in tf.trainable_variables()]
        
        pickle.dump( params, open(PARAMFILE, 'wb'))

if __name__ == '__main__':

    main()
