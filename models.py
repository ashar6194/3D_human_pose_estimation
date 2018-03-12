import convnet as cnn
import tensorflow as tf
import tflearn


class ModelRegression:

  def __init__(self, learning_rate=0.0001):
    input_ = tflearn.input_data(shape=[None, 840])
    linear = tflearn.fully_connected(input_, 1, activation='linear')
    regressor = tflearn.regression(linear, optimizer='sgd', loss='mean_square', metric='R2', learning_rate=learning_rate)
    with tf.variable_scope('regressor_object'):
      self.model = tflearn.DNN(regressor)

  def train_model(self, train_data, train_labels, n_epochs=10, show_metric=True):
    self.model.fit(train_data, train_labels, show_metric=show_metric, snapshot_epoch=False, n_epoch=n_epochs)

  def save_model(self, m):
    self.model.save(m)

  def load_model(self, m):
    self.model.load(m, weights_only=True)

  def predict_joints(self, training_features):
    return self.model.predict(training_features)

class Autoencoder:

  def __init__(self, n, strided=False, max_images=3):
    self.max_images = max_images
    self.n = n
    self.strided = strided

  def conv(self, x, channels_shape, name):
    return cnn.conv(x, [3, 3], channels_shape, 1, name)

  def conv2(self, x, channels_shape, name):
    return cnn.conv(x, [3, 3], channels_shape, 2, name)

  def deconv(self, x, channels_shape, name):
    return cnn.deconv(x, [3, 3], channels_shape, 1, name)

  def pool(self, x):
    return cnn.max_pool(x, 2, 2)

  def unpool(self, x):
    return cnn.unpool(x, 2)

  def resize_conv(self, x, channels_shape, name):
    shape = x.get_shape().as_list()
    height = shape[1] * 2
    width = shape[2] * 2
    resized = tf.image.resize_nearest_neighbor(x, [height, width])
    return cnn.conv(resized, [3, 3], channels_shape, 1, name, repad=True)

  def inference(self, images):
    if self.strided:
      return self.strided_inference(images)
    return self.inference_with_pooling(images)


class MiniAutoencoder(Autoencoder):
  def __init__(self, n, strided=True, max_images=3):
    Autoencoder.__init__(self, n, strided=strided, max_images=max_images)

  def strided_inference(self, images):
    tf.summary.image('input', images, max_outputs=self.max_images)
    with tf.variable_scope('encode1'):
      conv1 = self.conv(images, [3, 64], 'conv1_1')
      conv2 = self.conv2(conv1, [64, 64], 'conv1_2')

    with tf.variable_scope('encode2'):
      conv3 = self.conv(conv2, [64, 128], 'conv2_1')
      conv4 = self.conv2(conv3, [128, 128], 'conv2_2')

    with tf.variable_scope('encode3'):
      conv5 = self.conv(conv4, [128, 256], 'conv3_1')
      conv6 = self.conv(conv5, [256, 256], 'conv3_2')
      conv7 = self.conv2(conv6, [256, 256], 'conv3_3')

    with tf.variable_scope('decode1'):
      deconv7 = self.resize_conv(conv7, [256, 256], 'deconv3_3')
      deconv6 = self.deconv(deconv7, [256, 256], 'deconv3_2')
      deconv5 = self.deconv(deconv6, [128, 256], 'deconv3_1')

    with tf.variable_scope('decode2'):
      deconv4 = self.resize_conv(deconv5, [128, 128], 'deconv2_2')
      deconv3 = self.deconv(deconv4, [64, 128], 'deconv2_1')

    with tf.variable_scope('decode3'):
      deconv2 = self.resize_conv(deconv3, [64, 64], 'deconv1_2')
      deconv1 = self.deconv(deconv2, [self.n, 64], 'deconv1_1')

    # rgb_image = classifier.rgb(deconv1)
    # tf.summary.image('output', rgb_image, max_outputs=self.max_images)
    return deconv1  # , rgb_image


class SegNetAutoencoder(Autoencoder):
  def __init__(self, n, strided=False, max_images=3):
    Autoencoder.__init__(self, n, strided=strided, max_images=max_images)

  def inference_with_pooling(self, images):
    tf.summary.image('input', images, max_outputs=self.max_images)


    with tf.variable_scope('pool1'):
      conv1 = self.conv(images, [3, 64], 'conv1_1')
      conv2 = self.conv(conv1, [64, 64], 'conv1_2')
      pool1 = self.pool(conv2)

    with tf.variable_scope('pool2'):
      conv3 = self.conv(pool1, [64, 128], 'conv2_1')
      conv4 = self.conv(conv3, [128, 128], 'conv2_2')
      pool2 = self.pool(conv4)

    with tf.variable_scope('pool3'):
      conv5 = self.conv(pool2, [128, 256], 'conv3_1')
      conv6 = self.conv(conv5, [256, 256], 'conv3_2')
      conv7 = self.conv(conv6, [256, 256], 'conv3_3')
      pool3 = self.pool(conv7)

    with tf.variable_scope('pool4'):
      conv8 = self.conv(pool3, [256, 512], 'conv4_1')
      conv9 = self.conv(conv8, [512, 512], 'conv4_2')
      conv10 = self.conv(conv9, [512, 512], 'conv4_3')
      pool4 = self.pool(conv10)

    with tf.variable_scope('pool5'):
      conv11 = self.conv(pool4, [512, 512], 'conv5_1')
      conv12 = self.conv(conv11, [512, 512], 'conv5_2')
      conv13 = self.conv(conv12, [512, 512], 'conv5_3')
      pool5 = self.pool(conv13)

    with tf.variable_scope('unpool1'):
      unpool1 = self.unpool(pool5)
      deconv1 = self.deconv(unpool1, [512, 512], 'deconv5_3')
      deconv2 = self.deconv(deconv1, [512, 512], 'deconv5_2')
      deconv3 = self.deconv(deconv2, [512, 512], 'deconv5_1')
      deconv3_fuse = tf.add(deconv3, pool4)

    with tf.variable_scope('unpool2'):
      unpool2 = self.unpool(deconv3_fuse)
      deconv4 = self.deconv(unpool2, [512, 512], 'deconv4_3')
      deconv5 = self.deconv(deconv4, [512, 512], 'deconv4_2')
      deconv6 = self.deconv(deconv5, [256, 512], 'deconv4_1')
      deconv6_fuse = tf.add(deconv6, pool3)

    with tf.variable_scope('unpool3'):
      unpool3 = self.unpool(deconv6_fuse)
      deconv7 = self.deconv(unpool3, [256, 256], 'deconv3_3')
      deconv8 = self.deconv(deconv7, [256, 256], 'deconv3_2')
      deconv9 = self.deconv(deconv8, [128, 256], 'deconv3_1')
      deconv9_fuse = tf.add(deconv9, pool2)

    with tf.variable_scope('unpool4'):
      unpool4 = self.unpool(deconv9_fuse)
      deconv10 = self.deconv(unpool4, [128, 128], 'deconv2_2')
      deconv11 = self.deconv(deconv10, [64, 128], 'deconv2_1')

    with tf.variable_scope('unpool5'):
      unpool5 = self.unpool(deconv11)
      deconv12 = self.deconv(unpool5, [64, 64], 'deconv1_2')
      deconv13 = self.deconv(deconv12, [self.n, 64], 'deconv1_1')

    # rgb_image = classifier.rgb(deconv13)
    # tf.summary.image('output', rgb_image, max_outputs=self.max_images)
    return deconv13  # , rgb_image

  def strided_inference(self, images):
    tf.summary.image('input', images, max_outputs=self.max_images)

    with tf.variable_scope('pool1'):
      conv1 = self.conv(images, [3, 64], 'conv1_1')
      conv2 = self.conv2(conv1, [64, 64], 'conv1_2')

    with tf.variable_scope('pool2'):
      conv3 = self.conv(conv2, [64, 128], 'conv2_1')
      conv4 = self.conv2(conv3, [128, 128], 'conv2_2')

    with tf.variable_scope('pool3'):
      conv5 = self.conv(conv4, [128, 256], 'conv3_1')
      conv6 = self.conv(conv5, [256, 256], 'conv3_2')
      conv7 = self.conv2(conv6, [256, 256], 'conv3_3')

    with tf.variable_scope('pool4'):
      conv8 = self.conv(conv7, [256, 512], 'conv4_1')
      conv9 = self.conv(conv8, [512, 512], 'conv4_2')
      conv10 = self.conv2(conv9, [512, 512], 'conv4_3')

    with tf.variable_scope('pool5'):
      conv11 = self.conv(conv10, [512, 512], 'conv5_1')
      conv12 = self.conv(conv11, [512, 512], 'conv5_2')
      conv13 = self.conv2(conv12, [512, 512], 'conv5_3')

    with tf.variable_scope('unpool1'):
      deconv1 = self.resize_conv(conv13, [512, 512], 'deconv5_3')
      deconv2 = self.deconv(deconv1, [512, 512], 'deconv5_2')
      deconv3 = self.deconv(deconv2, [512, 512], 'deconv5_1')

    with tf.variable_scope('unpool2'):
      deconv4 = self.resize_conv(deconv3, [512, 512], 'deconv4_3')
      deconv5 = self.deconv(deconv4, [512, 512], 'deconv4_2')
      deconv6 = self.deconv(deconv5, [256, 512], 'deconv4_1')

    with tf.variable_scope('unpool3'):
      deconv7 = self.resize_conv(deconv6, [256, 256], 'deconv3_3')
      deconv8 = self.deconv(deconv7, [256, 256], 'deconv3_2')
      deconv9 = self.deconv(deconv8, [128, 256], 'deconv3_1')

    with tf.variable_scope('unpool4'):
      deconv10 = self.resize_conv(deconv9, [128, 128], 'deconv2_2')
      deconv11 = self.deconv(deconv10, [64, 128], 'deconv2_1')

    with tf.variable_scope('unpool5'):
      deconv12 = self.resize_conv(deconv11, [64, 64], 'deconv1_2')
      deconv13 = self.deconv(deconv12, [self.n, 64], 'deconv1_1')

    # rgb_image = classifier.rgb(deconv13)
    # tf.summary.image('output', rgb_image, max_outputs=self.max_images)
    return deconv13   # , rgb_image
