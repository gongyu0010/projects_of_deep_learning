# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 10:40:00 2019

@Author: Yu Gong
"""

import os

import sys

import tarfile

from six.moves import urllib

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")

FLAGS.data_dir = 'cifar10_data/'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

def maybe_download_and_extract():

  dest_directory = FLAGS.data_dir

  os.makedirs(dest_directory, exist_ok=True)

  filename = DATA_URL.split('/')[-1]

  filepath = os.path.join(dest_directory, filename)

  if not os.path.exists(filepath):

    def _progress(count, block_size, total_size):

      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,

          float(count * block_size) / float(total_size) * 100.0))

      sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)

    print()

    statinfo = os.stat(filepath)

    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

  extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')

  if not os.path.exists(extracted_dir_path):

    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

maybe_download_and_extract()