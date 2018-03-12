from PIL import Image
from scalar_ops import accuracy, per_class_accuracy
import cv2 as cv
import os
import glob
import numpy as np
import classifier
import config
import tensorflow as tf
import utils

tf.app.flags.DEFINE_integer('batch', 10, 'Batch size')
# tf.app.flags.DEFINE_string('subject', 1, 'test subject idx for model')
FLAGS = tf.app.flags.FLAGS


def test():

    if config.working_dataset == 'UBC_easy' or config.working_dataset == 'forced_UBC_easy':
      if config.working_dataset == 'UBC_easy':
        gt_folder = 'groundtruth'
      else:
        gt_folder = 'forced_groundtruth'
      images = tf.placeholder(tf.float32, shape=[224, 224, 4])
      image = tf.slice(images, [0, 0, 0], [224, 224, 3])
      alpha = tf.slice(images, [0, 0, 2], [224, 224, 1])
      alpha = tf.cast(alpha*255, tf.uint8)
      image /= 255
      image = tf.reshape(image, [-1, 224, 224, 3])

      labels = tf.placeholder(tf.float32, shape=[224, 224, 4])
      label = tf.slice(labels, [0, 0, 0], [224, 224, 3])
      label /= 255
      label = tf.reshape(label, [-1, 224, 224, 3])

      tf.summary.image('labels', label)
      one_hot_labels = classifier.one_hot(label)

      autoencoder = utils.get_autoencoder(config.autoencoder, config.working_dataset, config.strided)
      logits = autoencoder.inference(image)

      rgb_image = classifier.rgb(logits)
      tf.summary.image('output', rgb_image, max_outputs=3)

      rgb_image = tf.reshape(tf.cast(rgb_image * 255, tf.uint8), [224, 224, 3])
      rgba_image = tf.concat([rgb_image, alpha], 2)

      # Calculate Accuracy
      accuracy_op = accuracy(logits, one_hot_labels)
      tf.summary.scalar('accuracy', accuracy_op)

      pc_accuracy = per_class_accuracy(logits, one_hot_labels)
      pc_size = pc_accuracy.get_shape().as_list()
      for k in range(pc_size[0]):
        tf.summary.scalar('accuracy_class%02d' % (k + 1), pc_accuracy[k])
      tf.summary.tensor_summary('class_wise_accuracy', pc_accuracy)

      saver = tf.train.Saver(tf.global_variables())

      gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.gpu_memory_fraction)
      session_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

      with tf.Session(config=session_config) as sess:
        # File paths
        test_logs, ckpt_dir, main_dir, main_output_dir = config.get_directories(config.working_dataset)

        # Store Summaries
        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(test_logs)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        ckpt = tf.train.get_checkpoint_state(ckpt_dir)

        if not (ckpt and ckpt.model_checkpoint_path):
          print('No checkpoint file found')
          return

        ckpt_path = ckpt.model_checkpoint_path
        saver.restore(sess, ckpt_path)

        # Start Testing on the set
        train_set = 5
        cam_range = 3
        for num_test in range(1, train_set + 1):
          for num_cam in range(1, cam_range + 1):
            print 'Currently processing subject numbeer ' + str(num_test) + ' camera no- ' + str(num_cam)

            main_input_dir = '%s%d/images' % (main_dir, num_test)
            depth_file_list = glob.glob(os.path.join(main_input_dir, 'depthRender', 'Cam%d' % num_cam, '*.png'))
            label_file_list = glob.glob(os.path.join(main_input_dir, gt_folder, 'Cam%d' % num_cam, '*.png'))
            output_dir = '%s%d/Cam%d' % (main_output_dir, num_test, num_cam)
            if not os.path.exists(output_dir):
              os.makedirs(output_dir)

            for depth_file_name, label_file_name in zip(depth_file_list, label_file_list):
              output_file_name = os.path.join(output_dir, depth_file_name.split('/')[-1])

              if not os.path.exists(output_file_name):
                depth_image = Image.open(depth_file_name)
                label_image = Image.open(label_file_name)

                # Find Min-max non zero pixels in the label image
                label_image_array = np.array(label_image).sum(axis=2)

                itemidx = np.where(label_image_array.sum(axis=0) != 0)
                itemidy = np.where(label_image_array.sum(axis=1) != 0)

                # Crop and Resize Test Depth Image
                cropped_depth_image = depth_image.crop((min(itemidx[0]), min(itemidy[0]),
                                                        max(itemidx[0]), max(itemidy[0])))
                cropped_depth_image = cropped_depth_image.resize((224, 224), Image.NEAREST)

                # Crop and Resize Test Label Image
                cropped_label_image = label_image.crop((min(itemidx[0]), min(itemidy[0]),
                                                        max(itemidx[0]), max(itemidy[0])))
                cropped_label_image = cropped_label_image.resize((224, 224), Image.NEAREST)

                # Infer body-part labels from the learned model
                summary_str, inferred_image = sess.run([summary, rgba_image], feed_dict={images: cropped_depth_image,
                                                                                         labels: cropped_label_image})
                # Restore the original size of the inferred label image
                inferred_image = Image.fromarray(inferred_image.astype('uint8'))

                # Reshape to original size and aspect ratio
                resized_inferred_image = inferred_image.resize((max(itemidx[0]) - min(itemidx[0]),
                                                                max(itemidy[0]) - min(itemidy[0])), Image.NEAREST)
                resized_inferred_image = np.array(resized_inferred_image)
                Pad = np.zeros((424, 512, 4))
                Pad[min(itemidy[0]):max(itemidy[0]), min(itemidx[0]):max(itemidx[0]), :] = resized_inferred_image
                resized_inferred_image = Image.fromarray(Pad.astype('uint8'))

                # Save the restored image
                resized_inferred_image.save(output_file_name)

                # Write the summary collected from the session
                summary_writer.add_summary(summary_str)
                summary_writer.flush()

        coord.request_stop()
        coord.join(threads)

    elif config.working_dataset == 'MHAD' or config.working_dataset == 'UBC_MHAD':
        images = tf.placeholder(tf.float32, shape=[224, 224, 3])
        image = tf.slice(images, [0, 0, 0], [224, 224, 3])
        image /= 255
        image = tf.reshape(image, [-1, 224, 224, 3])

        autoencoder = utils.get_autoencoder(config.autoencoder, config.working_dataset, config.strided)
        logits = autoencoder.inference(image)

        rgb_image = classifier.rgb(logits)
        tf.summary.image('output', rgb_image, max_outputs=3)

        rgb_image = tf.reshape(tf.cast(rgb_image * 255, tf.uint8), [224, 224, 3])

        saver = tf.train.Saver(tf.global_variables())
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.gpu_memory_fraction)
        session_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        with tf.Session(config=session_config) as sess:

          # File paths
          test_logs, ckpt_dir, main_input_dir, output_sub_dir = config.get_directories(config.working_dataset)

          # Store Summaries and Initialize threads
          summary = tf.summary.merge_all()
          summary_writer = tf.summary.FileWriter(test_logs)
          coord = tf.train.Coordinator()
          threads = tf.train.start_queue_runners(sess=sess, coord=coord)

          ckpt = tf.train.get_checkpoint_state(ckpt_dir)

          if not (ckpt and ckpt.model_checkpoint_path):
            print('No checkpoint file found')
            return

          ckpt_path = ckpt.model_checkpoint_path
          saver.restore(sess, ckpt_path)
          subjects = 12
          cameras = 2
          actions = 11
          recordings = 5

          for cam in range(1, cameras + 1):
            for sub in range(1, subjects + 1):
              for act in range(1, actions + 1):
                for rec in range(1, recordings + 1):
                  print 'Currently processing subject no- ' + str(sub) + ' action no- ' + str(act) + ' recording no = ' + str(rec)
                  depth_file_list = glob.glob(os.path.join(main_input_dir, 'Kin%02d/S%02d/A%02d/R%02d/depth_png'
                                                           % (cam, sub, act, rec), '*.png'))
                  output_dir = os.path.join(main_input_dir, output_sub_dir, 'Kin%02d/S%02d/A%02d/R%02d/' % (cam, sub, act, rec))
                  if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                  for depth_file_name in depth_file_list:
                    output_file_name = os.path.join(output_dir, depth_file_name.split('/')[-1])

                    if not os.path.exists(output_file_name):
                      depth_image = cv.imread(depth_file_name)
                      depth_image_array = np.array(depth_image)

                      # Find Min-max non zero pixels in the label image
                      itemidx = np.where(depth_image_array.sum(axis=0) != 0)
                      itemidy = np.where(depth_image_array.sum(axis=1) != 0)
                      # print itemidx

                      # Crop and Resize Test Depth Image
                      try:
                        cropped_image = depth_image_array[min(itemidy[0]):max(itemidy[0]), min(itemidx[0]):max(itemidx[0]), :]
                        resized_depth_image = cv.resize(cropped_image, (224, 224), interpolation=cv.INTER_LINEAR)

                      # Infer body-part labels from the learned model

                        summary_str, inferred_image = sess.run([summary, rgb_image],
                                                             feed_dict={images: resized_depth_image})

                        # Reshape to original size and aspect ratio
                        resized_inferred_image = cv.resize(inferred_image,
                                                           ((max(itemidx[0]) - min(itemidx[0])),
                                                            (max(itemidy[0]) - min(itemidy[0]))),
                                                           interpolation=cv.INTER_NEAREST)

                        resized_inferred_image = np.array(resized_inferred_image)
                        Pad = np.zeros((480, 640, 3))
                        Pad[min(itemidy[0]):max(itemidy[0]), min(itemidx[0]):max(itemidx[0]), :] = resized_inferred_image
                        cv.imwrite(output_file_name, Pad[..., :: -1])

                        # Write the summary collected from the session
                        summary_writer.add_summary(summary_str)
                        summary_writer.flush()
                      except:
                        print 'Skipping the image name %s' % depth_file_name
          coord.request_stop()
          coord.join(threads)


def main(argv=None):
  # utils.restore_logs(FLAGS.test_logs)
  test()


if __name__ == '__main__':
  tf.app.run()
