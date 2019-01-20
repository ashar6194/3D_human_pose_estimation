from datetime import datetime
import os
import sys
import threading
import json
import numpy as np
import tensorflow as tf
from scipy import misc


tf.app.flags.DEFINE_string('labels_file', 'labels', 'Labels file')
tf.app.flags.DEFINE_string('output', 'input/UBC_easy/TFrecords/', 'Output data directory')
tf.app.flags.DEFINE_string('output_trn', 'Train_check', 'Output data Train Sub-directory')
tf.app.flags.DEFINE_string('output_val', 'Val', 'Output data Validation Sub-directory')
tf.app.flags.DEFINE_string('output_tst', 'Test', 'Output data Testing Sub-directory')
tf.app.flags.DEFINE_boolean('encode_train_images', True, 'Encode val images')
tf.app.flags.DEFINE_boolean('encode_val_images', False, 'Encode val images')
tf.app.flags.DEFINE_boolean('encode_test_images', False, 'Encode test images')


FLAGS = tf.app.flags.FLAGS
IGNORE_FILENAMES = ['.DS_Store']

def main(unused_argv):

  train_set = 10
  test_set = 1
  val_set = 1

  cam_range = 2
  mydict = {'train':[], 'train_label': [], 'val': [], 'val_label': [], 'test':[], 'test_label': []}
  errordict = {'train': [], 'train_label': [], 'val': [], 'val_label': [], 'test': [], 'test_label': []}

  if FLAGS.encode_train_images:
      for a in range(1, train_set + 1):
          for b in range(1, cam_range + 1):
              strname = 'train%d_Cam%d' % (a, b)
              op_file = os.path.join(FLAGS.output, FLAGS.output_trn, '%s.tfrecords' % strname)
              try:
                  mydict['train'].append(op_file)
              except:
                  errordict['train'].append(op_file)
              print ('Converting Train Example Number: %d' % a)

  print mydict
  print ('\n')
  print errordict
  with open('src/data_train2.json', 'w') as outfile:
      json.dump(mydict, outfile)
  with open('src/exception_train2.json', 'w') as outfile2:
      json.dump(errordict, outfile2)

if __name__ == '__main__':
  tf.app.run()