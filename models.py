# import classifier as classifier
import convnet as cnn
import tensorflow as tf
import tensorflow.contrib.slim as slim

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
    tf.summary.image('input', images[...,0:3], max_outputs=self.max_images)

    with tf.variable_scope('encode1'):
      conv1 = self.conv(images, [images.shape[3], 64], 'conv1_1')
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

    return deconv1

class MiniUNet(Autoencoder):
  def __init__(self, n, strided=True, max_images=3):
    Autoencoder.__init__(self, n, strided=strided, max_images=max_images)

  def strided_inference(self, images):
    tf.summary.image('input', images[..., 0:3], max_outputs=self.max_images)

    with tf.variable_scope('encode1'):
      conv1 = self.conv(images, [images.shape[3], 64], 'conv1_1')
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
      concat_conv4 = tf.concat([deconv5, conv4], axis=3)

    with tf.variable_scope('decode2'):
      deconv4 = self.resize_conv(concat_conv4, [256, 256], 'deconv2_2')
      deconv3 = self.deconv(deconv4, [64, 256], 'deconv2_1')
      concat_conv2 = tf.concat([deconv3, conv2], axis=3)

    with tf.variable_scope('decode3'):
      deconv2 = self.resize_conv(concat_conv2, [128, 128], 'deconv1_2')
      deconv1 = self.deconv(deconv2, [self.n, 128], 'deconv1_1')

    # rgb_image = classifier.rgb(deconv1)
    # tf.summary.image('output', rgb_image, max_outputs=self.max_images)
    return deconv1

class MiniResSum(Autoencoder):
  def __init__(self, n, strided=True, max_images=3):
    Autoencoder.__init__(self, n, strided=strided, max_images=max_images)

  def strided_inference(self, images):
    tf.summary.image('input', images[..., 0:3], max_outputs=self.max_images)

    with tf.variable_scope('encode1'):
      conv1 = self.conv(images, [images.shape[3], 64], 'conv1_1')
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
      sum_conv4 = tf.add(deconv5, conv4)

    with tf.variable_scope('decode2'):
      deconv4 = self.resize_conv(sum_conv4 , [128, 128], 'deconv2_2')
      deconv3 = self.deconv(deconv4, [64, 128], 'deconv2_1')
      sum_conv2 = tf.add(deconv3, conv2)

    with tf.variable_scope('decode3'):
      deconv2 = self.resize_conv(sum_conv2, [64, 64], 'deconv1_2')
      deconv1 = self.deconv(deconv2, [self.n, 64], 'deconv1_1')

    # rgb_image = classifier.rgb(deconv1)
    # tf.summary.image('output', rgb_image, max_outputs=self.max_images)
    return deconv1

class SegNetAutoencoder(Autoencoder):
  def __init__(self, n, strided=True, max_images=3):
    Autoencoder.__init__(self, n, strided=strided, max_images=max_images)

  def strided_inference(self, images):
    tf.summary.image('input', images[..., 0:3], max_outputs=self.max_images)

    with tf.variable_scope('pool1'):
      conv1 = self.conv(images, [images.shape[3], 64], 'conv1_1')
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
    return deconv13


class MiniFusion(Autoencoder):
  def __init__(self, n, strided=True, max_images=3):
    Autoencoder.__init__(self, n, strided=strided, max_images=max_images)

  def strided_inference(self, images):
    tf.summary.image('input', images[...,0:3], max_outputs=self.max_images)

    with tf.variable_scope('encode1'):
      conv1 = self.conv(images, [images.shape[3], 32], 'conv1_1')
      conv2 = self.conv2(conv1, [32, 32], 'conv1_2')

    with tf.variable_scope('encode2'):
      conv3 = self.conv(conv2, [32, 64], 'conv2_1')
      conv4 = self.conv2(conv3, [64, 64], 'conv2_2')

    with tf.variable_scope('encode3'):
      conv5 = self.conv(conv4, [64, 128], 'conv3_1')
      conv6 = self.conv(conv5, [128, 128], 'conv3_2')
      conv7 = self.conv2(conv6, [128, 128], 'conv3_3')

    with tf.variable_scope('decode1'):
      deconv7 = self.resize_conv(conv7, [128, 128], 'deconv3_3')
      deconv6 = self.deconv(deconv7, [128, 128], 'deconv3_2')
      deconv5 = self.deconv(deconv6, [64, 128], 'deconv3_1')

    with tf.variable_scope('decode2'):
      deconv4 = self.resize_conv(deconv5, [64, 64], 'deconv2_2')
      deconv3 = self.deconv(deconv4, [32, 64], 'deconv2_1')

    with tf.variable_scope('decode3'):
      deconv2 = self.resize_conv(deconv3, [32, 32], 'deconv1_2')
      deconv1 = self.deconv(deconv2, [self.n, 32], 'deconv1_1')

    return deconv1

class MiniFusionNet(Autoencoder):
  def __init__(self, n, strided=True, max_images=3):
    Autoencoder.__init__(self, n, strided=strided, max_images=max_images)

  def strided_inference(self, images):
    tf.summary.image('input', images[...,0:3], max_outputs=self.max_images)

    with tf.variable_scope('encode1_d'):
      conv1_d = self.conv(images[...,3:6], [3, 32], 'conv1_1')
      conv2_d = self.conv2(conv1_d, [32, 32], 'conv1_2')

    with tf.variable_scope('encode2_d'):
      conv3_d = self.conv(conv2_d, [32, 64], 'conv2_1')
      conv4_d = self.conv2(conv3_d, [64, 64], 'conv2_2')

    with tf.variable_scope('encode3_d'):
      conv5_d = self.conv(conv4_d, [64, 128], 'conv3_1')
      conv6_d = self.conv(conv5_d, [128, 128], 'conv3_2')
      conv7_d = self.conv2(conv6_d, [128, 128], 'conv3_3')

    with tf.variable_scope('encode1'):
      conv1 = self.conv(images[...,0:3], [3, 32], 'conv1_1')
      conv2 = self.conv2(conv1, [32, 32], 'conv1_2')
      conv2_fuse = tf.add(conv2, conv2_d)
      conv2 = tf.concat([conv2, conv2_fuse], axis=3)

    with tf.variable_scope('encode2'):
      conv3 = self.conv(conv2, [64, 64], 'conv2_1')
      conv4 = self.conv2(conv3, [64, 64], 'conv2_2')
      print conv4.shape, conv4_d.shape
      conv4_fuse = tf.add(conv4, conv4_d)
      conv4 = tf.concat([conv4, conv4_fuse], axis=3)

    with tf.variable_scope('encode3'):
      conv5 = self.conv(conv4, [128, 128], 'conv3_1')
      conv6 = self.conv(conv5, [128, 128], 'conv3_2')
      conv7 = self.conv2(conv6, [128, 128], 'conv3_3')
      conv7_fuse = tf.add(conv7, conv7_d)
      conv7 = tf.concat([conv7, conv7_fuse], axis=3)

    with tf.variable_scope('decode1'):
      deconv7 = self.resize_conv(conv7, [256, 256], 'deconv3_3')
      deconv6 = self.deconv(deconv7, [128, 256], 'deconv3_2')
      deconv5 = self.deconv(deconv6, [64, 128], 'deconv3_1')

    with tf.variable_scope('decode2'):
      deconv4 = self.resize_conv(deconv5, [64, 64], 'deconv2_2')
      deconv3 = self.deconv(deconv4, [32, 64], 'deconv2_1')

    with tf.variable_scope('decode3'):
      deconv2 = self.resize_conv(deconv3, [32, 32], 'deconv1_2')
      deconv1 = self.deconv(deconv2, [self.n, 32], 'deconv1_1')

    return deconv1


class MobileUnet(Autoencoder):
  """docstring for MobileUnet"""
  def __init__(self, n, strided=True, max_images=3):
    Autoencoder.__init__(self, n, strided=strided, max_images=max_images)

  def Convblock(self, inputs, num_filters, kernel=[1, 1]):
    net = slim.conv2d(inputs, num_filters, kernel_size=kernel, activation_fn=None)
    net = tf.nn.relu(slim.batch_norm(net, fused=True))
    return net

  def dwscb(self, inputs, num_filters, kernel=[3, 3]):
    net = slim.separable_convolution2d(inputs, num_outputs=None, depth_multiplier=1, kernel_size=kernel, activation_fn=None)
    net = tf.nn.relu(slim.batch_norm(net, fused=True))
    net = self.Convblock(net, num_filters)
    return net

  def ConvTRBlock(self, inputs, num_filters, kernel=[3, 3]):
    net = slim.conv2d_transpose(inputs, num_filters, kernel_size=kernel, stride=[2, 2], activation_fn=None)
    net = tf.nn.relu(slim.batch_norm(net, fused=True))
    return net

  def strided_inference(self, images):
    tf.summary.image('input', images[...,0:3], max_outputs=self.max_images)

    with tf.variable_scope('encode1'):
      net = self.dwscb(images, 64)
      net = self.dwscb(net, 64)
      net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')

    with tf.variable_scope('encode2'):
      net = self.dwscb(net, 128)
      net = self.dwscb(net, 128)
      net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')

    with tf.variable_scope('encode3'):
      net = self.dwscb(net, 256)
      net = self.dwscb(net, 256)
      net = self.dwscb(net, 256)
      net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')

    with tf.variable_scope('decode3'):
      net = self.ConvTRBlock(net, 256)
      net = self.dwscb(net, 256)
      net = self.dwscb(net, 256)
      net = self.dwscb(net, 256)

    with tf.variable_scope('decode2'):
      net = self.ConvTRBlock(net, 128)
      net = self.dwscb(net, 128)
      net = self.dwscb(net, 128)
      net = self.dwscb(net, 128)

    with tf.variable_scope('decode1'):
      net = self.ConvTRBlock(net, 64)
      net = self.dwscb(net, 64)
      net = self.dwscb(net, self.n)

    return net
