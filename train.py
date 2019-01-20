from inputs import inputs
from config import get_csv_filename
from scalar_ops import accuracy, loss, per_class_accuracy
import os
import csv_processor
import classifier
import config
import tensorflow as tf
import utils
from tqdm import *

tf.app.flags.DEFINE_integer('random_split_mode', False, 'train_test_split mode')
tf.app.flags.DEFINE_string('dataset_name', 'UBC_easy', 'To select name of the dataset being used')
tf.app.flags.DEFINE_integer('test_subject', 1, 'test subject index')
tf.app.flags.DEFINE_integer('summary_step', 100, 'Number of iterations before serializing log data')
tf.app.flags.DEFINE_integer('batch', 10, 'Batch size')
tf.app.flags.DEFINE_integer('steps', 15000, 'Number of training iterations')
FLAGS = tf.app.flags.FLAGS


def train(train_filename, train_ckpt, train_logs, ckpt_dir):
  images, labels = inputs(FLAGS.batch, train_filename)
  tf.summary.image('labels', labels)
  one_hot_labels = classifier.one_hot(labels)
  autoencoder = utils.get_autoencoder(config.autoencoder, config.working_dataset, config.strided)
  logits = autoencoder.inference(images)

  # Store output images
  rgb_image = classifier.rgb(logits)
  tf.summary.image('output', rgb_image, max_outputs=3)

  # Loss/Accuracy Metrics
  accuracy_op = accuracy(logits, one_hot_labels)
  loss_op = loss(logits, one_hot_labels)

  pc_accuracy = per_class_accuracy(logits, one_hot_labels)
  pc_size = pc_accuracy.get_shape().as_list()
  for k in range(pc_size[0]):
    tf.summary.scalar('accuracy_class%02d' % (k + 1), pc_accuracy[k])
  tf.summary.tensor_summary('class_wise_accuracy', pc_accuracy)
  tf.summary.scalar('accuracy', accuracy_op)
  tf.summary.scalar(loss_op.op.name, loss_op)
  optimizer = tf.train.AdamOptimizer(1e-04)
  train_step = optimizer.minimize(loss_op)
  init = tf.global_variables_initializer()
  saver = tf.train.Saver()
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.gpu_memory_fraction)
  session_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

  with tf.Session(config=session_config) as sess:
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if not ckpt:
      print('No checkpoint file present. Initializing...')
      global_step = 0
      sess.run(init)

    else:
      saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
      global_step = int(tf.train.latest_checkpoint(ckpt_dir).split('/')[-1].split('-')[-1])
      print 'Restoring the training from step- ' + str(global_step)

    summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(train_logs, sess.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in tqdm(range(global_step, FLAGS.steps + 1)):
      sess.run(train_step)

      if step % FLAGS.summary_step == 0:
        summary_str = sess.run(summary)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

        print '\ntraining step ' + str(step)
        print('\nAccuracy=  %06f , Loss=   %06f' % (accuracy_op.eval(), loss_op.eval()))

      if step == 0:
        saver.save(sess, train_ckpt, global_step=step)
      elif step == 100 or step == 500 or step == 1000 or step == 5000 or step == 7500 or step == 10000 \
          or step == 15000 or step == FLAGS.steps:
        saver.save(sess, train_ckpt, global_step=step, write_meta_graph=False)
    coord.request_stop()
    coord.join(threads)


def main(argv = None):

    # Directories for logs and checkpoints
    if FLAGS.dataset_name == 'MHAD':

      if FLAGS.random_split_mode:
        log_directory, ckpt_directory = config.get_directories_training(FLAGS.dataset_name)
        csv_filename = get_csv_filename(FLAGS.dataset_name)
        csv_dir = './csv'
        tfr_list, __, __ = csv_processor.get_random_percent_split(csv_filename, csv_dir, ratio=0.75)

        train_ckpt = '%smodel.ckpt' % ckpt_directory

        utils.restore_logs(log_directory)
        train(tfr_list, train_ckpt, log_directory, ckpt_directory)

      else:
        log_directory, ckpt_directory = config.get_directories_training(FLAGS.dataset_name)
        log_directory = os.path.join(log_directory, str(FLAGS.test_subject))
        ckpt_directory = os.path.join(ckpt_directory, str(FLAGS.test_subject)) + '/'

        csv_filename = get_csv_filename(FLAGS.dataset_name)
        csv_dir = './csv'
        tfr_list, __, __, __ = csv_processor.get_subject_wise_split(csv_filename, csv_dir, FLAGS.test_subject)

        train_ckpt = '%smodel.ckpt' % ckpt_directory

        utils.restore_logs(log_directory)
        print 'Currently processing for subject number %d' % FLAGS.test_subject
        train(tfr_list, train_ckpt, log_directory, ckpt_directory)

    elif FLAGS.dataset_name == 'UBC_easy' or FLAGS.dataset_name == 'forced_UBC_easy' \
        or FLAGS.dataset_name == 'UBC_MHAD' or FLAGS.dataset_name == 'UBC_interpolated' \
        or FLAGS.dataset_name == 'UBC_medium' or FLAGS.dataset_name == 'UBC_hard':

      log_directory, ckpt_directory = config.get_directories_training(FLAGS.dataset_name)
      csv_filename = get_csv_filename(FLAGS.dataset_name)
      tfr_list = csv_processor.get_train_list(csv_filename)

      train_ckpt = '%smodel.ckpt' % ckpt_directory

      utils.restore_logs(log_directory)
      train(tfr_list, train_ckpt, log_directory, ckpt_directory)


if __name__ == '__main__':
  tf.app.run()
