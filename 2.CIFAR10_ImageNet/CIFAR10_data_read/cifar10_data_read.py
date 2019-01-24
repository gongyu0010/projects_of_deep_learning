# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 10:50:00 2019

@Author: Yu Gong
"""

import os

import cifar10_input

from glob import glob

import tensorflow as tf

import PIL

from scipy.misc import imsave

def input_util(data_dir):

    file_list = glob(os.path.join(data_dir, '*.bin'))

    for file in file_list:

        if not tf.gfile.Exists(file):

            raise ValueError('Failed to find file: {}'.format(file))

    filename_queue = tf.train.string_input_producer(file_list)

    read_input = cifar10_input.read_cifar10(filename_queue=filename_queue)

    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    return reshaped_image


if __name__ == '__main__':

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        data_dir = '/Users/yugong/Documents/GitHub/projects_of_deep_learning/2.CIFAR10_ImageNet/CIFAR10_data_download/cifar10_data/cifar-10-batches-bin'

        reshaped_image = input_util(data_dir)

        threads = tf.train.start_queue_runners(sess=sess)

        os.makedirs(os.path.join(data_dir, 'raw'), exist_ok=True)

        save_dir = os.path.join(data_dir, 'raw')

        for i in range(30):

            image_array = sess.run(reshaped_image)

            imsave(save_dir + '/{}.jpg'.format(i), image_array)

