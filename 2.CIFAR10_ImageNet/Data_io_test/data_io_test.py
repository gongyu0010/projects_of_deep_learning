# -*- coding: utf-8 -*-
"""
Created on We Jan 23 10:00:00 2019

@Author: Yu Gong
"""


import os

import tensorflow as tf

os.makedirs('read', exist_ok=True)

with tf.Session() as sess:

    filename = ['A.jpg', 'B.jpg', 'C.jpg']

    filename_queue = tf.train.string_input_producer(filename, num_epochs=5, shuffle=False)

    reader = tf.WholeFileReader()

    key, value = reader.read(filename_queue)

    tf.local_variables_initializer().run()

    threads = tf.train.start_queue_runners(sess)

    i = 0

    while True:

        i += 1

        image_data = sess.run(value)

        with open('read/test_{}.jpg'.format(i), 'wb') as file:

            file.write(image_data)