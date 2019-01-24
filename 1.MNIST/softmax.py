# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 09:00:00 2019

@Author: Yu Gong
"""

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


x = tf.placeholder(tf.float32, shape=[None, 784])

W = tf.Variable(tf.zeros(shape=[784, 10]))

b = tf.Variable(tf.zeros(shape=[10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, shape=[None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    print('Train Start !!!')

    for iter in range(1000):

        batch_xs, batch_ys = mnist.train.next_batch(128)

        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print('\n', sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

    print('\nTrain Finish !!!')





