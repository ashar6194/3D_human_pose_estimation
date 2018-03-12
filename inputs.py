import tensorflow as tf


def read_and_decode_single_example(filename):
  # print ('The file name is as follows\n\n %s\n\n' % filename)
  filename_queue = tf.train.string_input_producer(filename, num_epochs=None)

  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)

  features = tf.parse_single_example(
    serialized_example,
    features={
      'image/encoded': tf.FixedLenFeature([], tf.string),
      'image/label_pixels': tf.FixedLenFeature([], tf.string)
    })

  # Check for the images and the labels in one tfrecords file
  image = features['image/encoded']
  image = tf.cast(tf.image.decode_png(image, 3), tf.float32)
  image = tf.image.resize_images(image, [224, 224])
  image /= 255
  image.set_shape([224, 224, 3])

  label = features['image/label_pixels']
  label = tf.cast(tf.image.decode_png(label, 3), tf.float32)
  label = tf.image.resize_images(label, [224, 224])
  label /= 255
  label.set_shape([224, 224, 3])
  return image, label


def inputs(batch_size, train_filename):
  image, label = read_and_decode_single_example(train_filename)
  image_batch, label_batch = tf.train.shuffle_batch(
      [image, label],
      batch_size=batch_size,
      capacity=2000, min_after_dequeue=1000)
  return image_batch, label_batch
