from PIL import Image
from datetime import datetime
import os
import sys
import threading
import json
import numpy as np
import tensorflow as tf
import config

tf.app.flags.DEFINE_string('dataset', 'UBC_hard', 'Output data Train Sub-directory')
tf.app.flags.DEFINE_string('output_trn', 'Train', 'Output data Train Sub-directory')
tf.app.flags.DEFINE_integer('train_shards', 1, 'Number of shards in training TFRecord files')
tf.app.flags.DEFINE_integer('test_shards', 1, 'Number of shards in test TFRecord files')
tf.app.flags.DEFINE_integer('threads', 1, 'Number of threads to pre-process the images')

FLAGS = tf.app.flags.FLAGS
IGNORE_FILENAMES = ['.DS_Store']


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, image_buffer_lbl, height, width):
  example = tf.train.Example(features=tf.train.Features(feature={
    'image/encoded': _bytes_feature(image_buffer),
    'image/label_pixels': _bytes_feature(image_buffer_lbl)
  }))
  return example


class ImageCoder(object):
  def __init__(self):
    self._sess = tf.Session()
    self._png_data = tf.placeholder(dtype=tf.string)
    self._decode_png = tf.image.decode_png(self._png_data, channels=3)

  def decode_png(self, image_data):
    image = self._sess.run(self._decode_png, feed_dict={self._png_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _process_image(filename, coder, flabel):
    input_image = Image.open(filename)
    input_image_z = np.array(input_image).sum(axis=2)
    itemidx = np.where(input_image_z.sum(axis=0) != 0)
    itemidy = np.where(input_image_z.sum(axis=1) != 0)
    cropped_image = input_image.crop((min(itemidx[0]), min(itemidy[0]),
                                     max(itemidx[0]), max(itemidy[0])))

    if flabel:
        resized_image = cropped_image.resize((224, 224), Image.NEAREST)
    else:
        resized_image = cropped_image.resize((224, 224), Image.BILINEAR)

    file_new = 'temp.png'
    resized_image.save(file_new)
    # print filename
    with tf.gfile.FastGFile(file_new, 'r') as f:
        image_data = f.read()
    image = coder.decode_png(image_data)

    assert len(image.shape) == 3
    assert image.shape[2] == 3
    height, width, _ = image.shape
    return image_data, height, width


def _process_image_files_batch(coder, thread_index, ranges, name, filenames, filenames_lbl,
                               output_dir, num_shards):
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  for s in range(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    if num_shards == 1:
      output_filename = '%s.tfrecords' % name
    else:
      output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
    output_file = os.path.join(output_dir, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in files_in_shard:
      filename = filenames[i]
      if filename.split('/')[-1] in IGNORE_FILENAMES:
        continue
      image_buffer, height, width = _process_image(filename, coder, False)

      filename_lbl = filenames_lbl[i]
      if filename_lbl.split('/')[-1] in IGNORE_FILENAMES:
        continue
      image_buffer_lbl, height, width = _process_image(filename_lbl, coder, True)

      example = _convert_to_example(filename, image_buffer, image_buffer_lbl, height, width)
      writer.write(example.SerializeToString())
      shard_counter += 1
      counter += 1

      if not counter % 1000:
        sys.stdout.flush()
    writer.close()
    sys.stdout.flush()
    shard_counter = 0

  sys.stdout.flush()


def _process_image_files(name, filenames, filenames_lbl, output_dir, num_shards):
  spacing = np.linspace(0, len(filenames), FLAGS.threads + 1).astype(np.int)
  ranges = []

  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i+1]])
  sys.stdout.flush()
  coord = tf.train.Coordinator()
  coder = ImageCoder()

  threads = []
  for thread_index in range(len(ranges)):
    args = (coder, thread_index, ranges, name, filenames, filenames_lbl, output_dir, num_shards)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)
  coord.join(threads)
  print('%s: Finished writing all %d images in data set.' %
        (datetime.now(), len(filenames)))

  sys.stdout.flush()


def _process_dataset(name, directory, directory_lbl, output_dir, num_shards):
  file_path = '%s/*' % directory
  file_path2 = '%s/*' % directory_lbl
  filenames = tf.gfile.Glob(file_path)
  filenames_lbl = tf.gfile.Glob(file_path2)
  _process_image_files(name, filenames, filenames_lbl, output_dir, num_shards)


def _train_process(main_dir, output_directory):
  assert not FLAGS.train_shards % FLAGS.threads, 'Please make the FLAGS.threads commensurate with FLAGS.train_shards'
  assert not FLAGS.test_shards % FLAGS.threads, 'Please make the FLAGS.threads commensurate with FLAGS.test_shards'
  print('Saving results to %s' % main_dir)
  if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# parser for UBC easy data
  if FLAGS.dataset == 'UBC_easy' or FLAGS.dataset == 'UBC_medium' or FLAGS.dataset == 'UBC_hard':

    if FLAGS.dataset == 'UBC_hard':
      train_set = 60
      valid_set = 19
    else:
      train_set = 180
      valid_set = 60
    cam_range = 3
    path_dictionary = {'train': [], 'val': [], 'test': []}
    error_dictionary = {'train': [], 'val': [], 'test': []}
    # for a in range(1, train_set + 1):
        # for b in range(1, cam_range + 1):
        #     strname = 'train%d_Cam%d' % (a, b)
        #     strdir = os.path.join(main_dir, 'train/%d/images/depthRender/Cam%d' % (a, b))
        #     strdir_lbl = os.path.join(main_dir, 'train/%d/images/groundtruth/Cam%d' % (a, b))
        #     op_file = os.path.join(output_directory, '%s.tfrecords' % strname)
        #     try:
        #         _process_dataset(strname, strdir, strdir_lbl,  output_directory, FLAGS.train_shards)
        #         path_dictionary['train'].append(op_file)
        #     except:
        #         error_dictionary['train'].append(op_file)
        #     print ('Converting Train Example Number: %d, Camera %d' % (a, b))

    for a in range(1, valid_set + 1):
      for b in range(1, cam_range + 1):
        strname = 'valid%d_Cam%d' % (a, b)
        strdir = os.path.join(main_dir, 'valid/%d/images/depthRender/Cam%d' % (a, b))
        strdir_lbl = os.path.join(main_dir, 'valid/%d/images/groundtruth/Cam%d' % (a, b))
        op_file = os.path.join(output_directory, '%s.tfrecords' % strname)
        try:
          _process_dataset(strname, strdir, strdir_lbl, output_directory, FLAGS.train_shards)
          path_dictionary['val'].append(op_file)
        except:
          error_dictionary['val'].append(op_file)
        print ('Converted Validation Example Number: %d, Camera %d' % (a, b))

  if FLAGS.dataset == 'forced_UBC_easy':
      train_set = 60
      valid_set = 19
      cam_range = 3
      path_dictionary = {'train': [], 'valid': [], 'test': []}
      error_dictionary = {'train': [], 'valid': [], 'test': []}
      for a in range(8, train_set + 1):
          for b in range(1, cam_range + 1):
              strname = 'train%d_Cam%d' % (a, b)
              strdir = os.path.join(main_dir, 'train/%d/images/depthRender/Cam%d' % (a, b))
              strdir_lbl = os.path.join(main_dir, 'train/%d/images/forced_groundtruth/Cam%d' % (a, b))
              op_file = os.path.join(output_directory, '%s.tfrecords' % strname)
              try:
                  _process_dataset(strname, strdir, strdir_lbl, output_directory, FLAGS.train_shards)
                  path_dictionary['train'].append(op_file)
              except:
                  error_dictionary['train'].append(op_file)
              print ('Converted Train Example Number: %d, Camera %d' % (a, b))

      for a in range(1, valid_set + 1):
          for b in range(1, cam_range + 1):
              strname = 'valid%d_Cam%d' % (a, b)
              strdir = os.path.join(main_dir, 'valid/%d/images/depthRender/Cam%d' % (a, b))
              strdir_lbl = os.path.join(main_dir, 'valid/%d/images/forced_groundtruth/Cam%d' % (a, b))
              op_file = os.path.join(output_directory, '%s.tfrecords' % strname)
              try:
                  _process_dataset(strname, strdir, strdir_lbl, output_directory, FLAGS.train_shards)
                  path_dictionary['valid'].append(op_file)
              except:
                  error_dictionary['valid'].append(op_file)
              print ('Converted Validation Example Number: %d, Camera %d' % (a, b))

      with open('training_data_forced_ubc_hard.json', 'a') as outfile:
          json.dump(path_dictionary, outfile)
      with open('exception_forced_ubc_hard.json', 'a') as outfile2:
          json.dump(error_dictionary, outfile2)

# parser for Berkeley data
  if FLAGS.dataset == 'MHAD':
      cameras = 2
      subjects = 3
      actions = 11
      recordings = 5

      path_dictionary = {'train': dict((k, []) for k in range(1, subjects + 1))}
      error_dictionary = {'train': dict((k, []) for k in range(1, subjects + 1))}

      for cam in range(2, cameras + 1):
          for sub in range(1, subjects + 1):
              for act in range(1, actions + 1):
                  for rec in range(1, recordings + 1):
                      strname = 'Kin%02d_s%02d_a%02d_r%02d' % (cam, sub, act, rec)
                      strdir = os.path.join(main_dir, 'Kin%02d/S%02d/A%02d/R%02d/depth_png' % (cam, sub, act, rec))
                      strdir_lbl = os.path.join(main_dir, 'Kin%02d/S%02d/A%02d/R%02d/labels' % (cam, sub, act, rec))
                      op_file = os.path.join(output_directory, '%s.tfrecords' % strname)
                      try:
                        _process_dataset(strname, strdir, strdir_lbl,  output_directory, FLAGS.train_shards)
                        path_dictionary['train'][sub].append(op_file)
                      except:
                        error_dictionary['train'][sub].append(op_file)
                      print ('Converting: Camera%02d, Subject%02d, Action%02d, Recording%02d ' % (cam, sub, act, rec))

      with open('training_mhad_fitted_planev2.json', 'w') as outfile:
          json.dump(path_dictionary, outfile)
      with open('error_mhad_fitted_planev2.json', 'w') as outfile2:
          json.dump(error_dictionary, outfile2)

  if FLAGS.dataset == 'MHAD_UBC':
      train_set = 60
      valid_set = 19
      cam_range = 3
      path_dictionary = {'train': [], 'valid': [], 'test': []}
      error_dictionary = {'train': [], 'valid': [], 'test': []}
      for a in range(1, train_set + 1):
          for b in range(1, cam_range + 1):
              strname = 'train%d_Cam%d' % (a, b)
              strdir = os.path.join(main_dir, 'train/%d/images/depthRender/Cam%d' % (a, b))
              strdir_lbl = os.path.join(main_dir, 'train/%d/images/interpol_groundtruth/Cam%d' % (a, b))
              op_file = os.path.join(output_directory, '%s.tfrecords' % strname)
              try:
                _process_dataset(strname, strdir, strdir_lbl, output_directory, FLAGS.train_shards)
                path_dictionary['train'].append(op_file)
                print ('Converted Train Example Number: %d, Camera %d' % (a, b))
              except:
                error_dictionary['train'].append(op_file)

      for a in range(1, valid_set + 1):
          for b in range(1, cam_range + 1):
              strname = 'valid%d_Cam%d' % (a, b)
              strdir = os.path.join(main_dir, 'valid/%d/images/depthRender/Cam%d' % (a, b))
              strdir_lbl = os.path.join(main_dir, 'valid/%d/images/interpol_groundtruth/Cam%d' % (a, b))
              op_file = os.path.join(output_directory, '%s.tfrecords' % strname)
              try:
                  _process_dataset(strname, strdir, strdir_lbl, output_directory, FLAGS.train_shards)
                  path_dictionary['valid'].append(op_file)
                  print ('Converted Train Example Number: %d, Camera %d' % (a, b))
              except:
                  error_dictionary['valid'].append(op_file)

      with open('./json/training_data_mhad_ubc_hard.json', 'a') as outfile:
          json.dump(path_dictionary, outfile)
      with open('./json/exception_mhad_ubc_hard.json', 'a') as outfile2:
          json.dump(error_dictionary, outfile2)


def main(unused_argv):
    main_dir_mhad = '/media/mcao/Miguel/MHAD_fitted_plane/Kinect/'
    main_dir_ubc = '/media/mcao/Miguel/input/UBC_easy/'

    print 'Running for the dataset  = ' + FLAGS.dataset

    if FLAGS.dataset == 'UBC_easy':
      main_dir = main_dir_ubc
      tfrecord_path = 'TFrecords'
      output_directory = os.path.join(main_dir, tfrecord_path, FLAGS.output_trn)
      _train_process(main_dir, output_directory)

    elif FLAGS.dataset == 'UBC_medium':
      main_dir_ubc = '/media/mcao/Miguel/UBC_medium/'
      main_dir = main_dir_ubc
      tfrecord_path = 'TFrecords'
      output_directory = os.path.join(main_dir, tfrecord_path, FLAGS.output_trn)
      _train_process(main_dir, output_directory)

    elif FLAGS.dataset == 'UBC_hard':
      main_dir_ubc = '/media/mcao/Miguel/UBC_hard/'
      main_dir = main_dir_ubc
      tfrecord_path = 'TFrecords'
      output_directory = os.path.join(main_dir, tfrecord_path, FLAGS.output_trn)
      _train_process(main_dir, output_directory)

    elif FLAGS.dataset == 'MHAD':
      main_dir = main_dir_mhad
      tfrecord_path = 'TFrecords_fittedplane'
      output_directory = os.path.join(main_dir, tfrecord_path, FLAGS.output_trn)
      _train_process(main_dir, output_directory)

    elif FLAGS.dataset == 'forced_UBC_easy':
      main_dir_ubc = '/media/mcao/Miguel/UBC_hard/'
      main_dir = main_dir_ubc
      tfrecord_path = 'Forced_TFrecords'
      output_directory = os.path.join(main_dir, tfrecord_path, FLAGS.output_trn)
      _train_process(main_dir, output_directory)

    elif FLAGS.dataset == 'UBC_interpolated':
      main_dir_ubc = '/media/mcao/Miguel/UBC_hard/'
      main_dir = main_dir_ubc
      tfrecord_path = 'MHAD_learned_TFrecords'
      output_directory = os.path.join(main_dir, tfrecord_path, FLAGS.output_trn)
      _train_process(main_dir, output_directory)


if __name__ == '__main__':
  tf.app.run()