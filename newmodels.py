import tensorflow as tf
import convnet as cnn
from tensorflow.contrib.layers.python.layers import initializers
from enet_utils import PReLU, spatial_dropout, max_unpool
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


class ENet_model(Autoencoder):

	def __init__(self, n, strided=True, max_images=3):
		Autoencoder.__init__(self, n, strided=strided, max_images=max_images)
		self.no_of_classes = n
		self.wd = 2e-4 # (weight decay)
		self.lr = 5e-4 # (learning rate)

    def strided_inference(self, images):

    	tf.summary.image('input', images[...,0:3], max_outputs=self.max_images)

        # encoder:
        # # initial block:
        network = self.initial_block(x=images[...,0:3], scope="inital")
        
        # # layer 1:
        # # # save the input shape to use in max_unpool in the decoder:
        inputs_shape_1 = network.get_shape().as_list()
        network, pooling_indices_1 = self.encoder_bottleneck_regular(x=network,
                    output_depth=64, drop_prob=self.early_drop_prob_ph,
                    scope="bottleneck_1_0", downsampling=True)
        

        network = self.encoder_bottleneck_regular(x=network, output_depth=64,
                    drop_prob=self.early_drop_prob_ph, scope="bottleneck_1_1")
        
        network = self.encoder_bottleneck_regular(x=network, output_depth=64,
                    drop_prob=self.early_drop_prob_ph, scope="bottleneck_1_2")
        
        network = self.encoder_bottleneck_regular(x=network, output_depth=64,
                    drop_prob=self.early_drop_prob_ph, scope="bottleneck_1_3")
        
        network = self.encoder_bottleneck_regular(x=network, output_depth=64,
                    drop_prob=self.early_drop_prob_ph, scope="bottleneck_1_4")
        

        # # layer 2:
        # # # save the input shape to use in max_unpool in the decoder:
        inputs_shape_2 = network.get_shape().as_list()
        network, pooling_indices_2 = self.encoder_bottleneck_regular(x=network,
                    output_depth=128, drop_prob=self.late_drop_prob_ph,
                    scope="bottleneck_2_0", downsampling=True)
        

        network = self.encoder_bottleneck_regular(x=network, output_depth=128,
                        drop_prob=self.late_drop_prob_ph, scope="bottleneck_2_1")
        
        network = self.encoder_bottleneck_dilated(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_2_2",
                    dilation_rate=2)
        
        network = self.encoder_bottleneck_asymmetric(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_2_3")
        
        network = self.encoder_bottleneck_dilated(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_2_4",
                    dilation_rate=4)
        
        network = self.encoder_bottleneck_regular(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_2_5")
        
        network = self.encoder_bottleneck_dilated(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_2_6",
                    dilation_rate=8)

        network = self.encoder_bottleneck_asymmetric(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_2_7")
        
        network = self.encoder_bottleneck_dilated(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_2_8",
                    dilation_rate=16)
        

        # layer 3:
        network = self.encoder_bottleneck_regular(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_1")
        
        network = self.encoder_bottleneck_dilated(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_2",
                    dilation_rate=2)
        
        network = self.encoder_bottleneck_asymmetric(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_3")

        network = self.encoder_bottleneck_dilated(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_4",
                    dilation_rate=4)
        
        network = self.encoder_bottleneck_regular(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_5")
        
        network = self.encoder_bottleneck_dilated(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_6",
                    dilation_rate=8)
        
        network = self.encoder_bottleneck_asymmetric(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_7")
        
        network = self.encoder_bottleneck_dilated(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_8",
                    dilation_rate=16)
        
        # decoder:
        # # layer 4:
        network = self.decoder_bottleneck(x=network, output_depth=64,
                    scope="bottleneck_4_0", upsampling=True,
                    pooling_indices=pooling_indices_2, output_shape=inputs_shape_2)
        
        network = self.decoder_bottleneck(x=network, output_depth=64,
                    scope="bottleneck_4_1")
        
        network = self.decoder_bottleneck(x=network, output_depth=64,
                    scope="bottleneck_4_2")
        
        # # layer 5:
        network = self.decoder_bottleneck(x=network, output_depth=16,
                    scope="bottleneck_5_0", upsampling=True,
                    pooling_indices=pooling_indices_1, output_shape=inputs_shape_1)

        network = self.decoder_bottleneck(x=network, output_depth=16,
                    scope="bottleneck_5_1")
        
        # fullconv:
        network = tf.contrib.slim.conv2d_transpose(network, self.no_of_classes,
                    [2, 2], stride=2, scope="fullconv", padding="SAME")
        

        self.logits = network

    def initial_block(self, x, scope):
        # convolution branch:
        W_conv = self.get_variable_weight_decay(scope + "/W",
                    shape=[3, 3, 3, 13], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="encoder_wd_losses")
        b_conv = self.get_variable_weight_decay(scope + "/b", shape=[13], # ([out_depth])
                    initializer=tf.constant_initializer(0),
                    loss_category="encoder_wd_losses")
        conv_branch = tf.nn.conv2d(x, W_conv, strides=[1, 2, 2, 1],
                    padding="SAME") + b_conv

        # max pooling branch:
        pool_branch = tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding="VALID")

        # concatenate the branches:
        concat = tf.concat([conv_branch, pool_branch], axis=3) # (3: the depth axis)

        # apply batch normalization and PReLU:
        output = tf.contrib.slim.batch_norm(concat)
        output = PReLU(output, scope=scope)

        return output

    def encoder_bottleneck_regular(self, x, output_depth, drop_prob, scope, proj_ratio=4, downsampling=False):
        input_shape = x.get_shape().as_list()
        input_depth = input_shape[3]

        internal_depth = int(output_depth/proj_ratio)

        # convolution branch:
        conv_branch = x

        # # 1x1 projection:
        if downsampling:
            W_conv = self.get_variable_weight_decay(scope + "/W_proj",
                        shape=[2, 2, input_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                        initializer=tf.contrib.layers.xavier_initializer(),
                        loss_category="encoder_wd_losses")
            conv_branch = tf.nn.conv2d(conv_branch, W_conv, strides=[1, 2, 2, 1],
                        padding="VALID") # NOTE! no bias terms
        else:
            W_proj = self.get_variable_weight_decay(scope + "/W_proj",
                        shape=[1, 1, input_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                        initializer=tf.contrib.layers.xavier_initializer(),
                        loss_category="encoder_wd_losses")
            conv_branch = tf.nn.conv2d(conv_branch, W_proj, strides=[1, 1, 1, 1],
                        padding="VALID") # NOTE! no bias terms
        # # # batch norm and PReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        conv_branch = PReLU(conv_branch, scope=scope + "/proj")

        # # conv:
        W_conv = self.get_variable_weight_decay(scope + "/W_conv",
                    shape=[3, 3, internal_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="encoder_wd_losses")
        b_conv = self.get_variable_weight_decay(scope + "/b_conv", shape=[internal_depth], # ([out_depth])
                    initializer=tf.constant_initializer(0),
                    loss_category="encoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_conv, strides=[1, 1, 1, 1],
                    padding="SAME") + b_conv
        # # # batch norm and PReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        conv_branch = PReLU(conv_branch, scope=scope + "/conv")

        # # 1x1 expansion:
        W_exp = self.get_variable_weight_decay(scope + "/W_exp",
                    shape=[1, 1, internal_depth, output_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="encoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_exp, strides=[1, 1, 1, 1],
                    padding="VALID") # NOTE! no bias terms
        # # # batch norm:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        # NOTE! no PReLU here

        # # regularizer:
        conv_branch = spatial_dropout(conv_branch, drop_prob)


        # main branch:
        main_branch = x

        if downsampling:
            # max pooling with argmax (for use in max_unpool in the decoder):
            main_branch, pooling_indices = tf.nn.max_pool_with_argmax(main_branch,
                        ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            # (everytime we downsample, we also increase the feature block depth)

            # pad with zeros so that the feature block depth matches:
            depth_to_pad = output_depth - input_depth
            paddings = tf.convert_to_tensor([[0, 0], [0, 0], [0, 0], [0, depth_to_pad]])
            # (paddings is an integer tensor of shape [4, 2] where 4 is the rank
            # of main_branch. For each dimension D (D = 0, 1, 2, 3) of main_branch,
            # paddings[D, 0] is the no of values to add before the contents of
            # main_branch in that dimension, and paddings[D, 0] is the no of
            # values to add after the contents of main_branch in that dimension)
            main_branch = tf.pad(main_branch, paddings=paddings, mode="CONSTANT")


        # add the branches:
        merged = conv_branch + main_branch

        # apply PReLU:
        output = PReLU(merged, scope=scope + "/output")

        if downsampling:
            return output, pooling_indices
        else:
            return output

    def encoder_bottleneck_dilated(self, x, output_depth, drop_prob, scope, dilation_rate, proj_ratio=4):
        input_shape = x.get_shape().as_list()
        input_depth = input_shape[3]

        internal_depth = int(output_depth/proj_ratio)

        # convolution branch:
        conv_branch = x

        # # 1x1 projection:
        W_proj = self.get_variable_weight_decay(scope + "/W_proj",
                    shape=[1, 1, input_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="encoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_proj, strides=[1, 1, 1, 1],
                    padding="VALID") # NOTE! no bias terms
        # # # batch norm and PReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        conv_branch = PReLU(conv_branch, scope=scope + "/proj")

        # # dilated conv:
        W_conv = self.get_variable_weight_decay(scope + "/W_conv",
                    shape=[3, 3, internal_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="encoder_wd_losses")
        b_conv = self.get_variable_weight_decay(scope + "/b_conv", shape=[internal_depth], # ([out_depth])
                    initializer=tf.constant_initializer(0),
                    loss_category="encoder_wd_losses")
        conv_branch = tf.nn.atrous_conv2d(conv_branch, W_conv, rate=dilation_rate,
                    padding="SAME") + b_conv
        # # # batch norm and PReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        conv_branch = PReLU(conv_branch, scope=scope + "/conv")

        # # 1x1 expansion:
        W_exp = self.get_variable_weight_decay(scope + "/W_exp",
                    shape=[1, 1, internal_depth, output_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="encoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_exp, strides=[1, 1, 1, 1],
                    padding="VALID") # NOTE! no bias terms
        # # # batch norm:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        # NOTE! no PReLU here

        # # regularizer:
        conv_branch = spatial_dropout(conv_branch, drop_prob)


        # main branch:
        main_branch = x


        # add the branches:
        merged = conv_branch + main_branch

        # apply PReLU:
        output = PReLU(merged, scope=scope + "/output")

        return output

    def encoder_bottleneck_asymmetric(self, x, output_depth, drop_prob, scope, proj_ratio=4):
        input_shape = x.get_shape().as_list()
        input_depth = input_shape[3]

        internal_depth = int(output_depth/proj_ratio)

        # convolution branch:
        conv_branch = x

        # # 1x1 projection:
        W_proj = self.get_variable_weight_decay(scope + "/W_proj",
                    shape=[1, 1, input_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="encoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_proj, strides=[1, 1, 1, 1],
                    padding="VALID") # NOTE! no bias terms
        # # # batch norm and PReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        conv_branch = PReLU(conv_branch, scope=scope + "/proj")

        # # asymmetric conv:
        # # # asymmetric conv 1:
        W_conv1 = self.get_variable_weight_decay(scope + "/W_conv1",
                    shape=[5, 1, internal_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="encoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_conv1, strides=[1, 1, 1, 1],
                    padding="SAME") # NOTE! no bias terms
        # # # asymmetric conv 2:
        W_conv2 = self.get_variable_weight_decay(scope + "/W_conv2",
                    shape=[1, 5, internal_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="encoder_wd_losses")
        b_conv2 = self.get_variable_weight_decay(scope + "/b_conv2", shape=[internal_depth], # ([out_depth])
                    initializer=tf.constant_initializer(0),
                    loss_category="encoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_conv2, strides=[1, 1, 1, 1],
                    padding="SAME") + b_conv2
        # # # batch norm and PReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        conv_branch = PReLU(conv_branch, scope=scope + "/conv")

        # # 1x1 expansion:
        W_exp = self.get_variable_weight_decay(scope + "/W_exp",
                    shape=[1, 1, internal_depth, output_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="encoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_exp, strides=[1, 1, 1, 1],
                    padding="VALID") # NOTE! no bias terms
        # # # batch norm:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        # NOTE! no PReLU here

        # # regularizer:
        conv_branch = spatial_dropout(conv_branch, drop_prob)


        # main branch:
        main_branch = x


        # add the branches:
        merged = conv_branch + main_branch

        # apply PReLU:
        output = PReLU(merged, scope=scope + "/output")

        return output

    def decoder_bottleneck(self, x, output_depth, scope, proj_ratio=4, upsampling=False, pooling_indices=None, output_shape=None):
        # NOTE! decoder uses ReLU instead of PReLU

        input_shape = x.get_shape().as_list()
        input_depth = input_shape[3]

        internal_depth = int(output_depth/proj_ratio)

        # main branch:
        main_branch = x

        if upsampling:
            # # 1x1 projection (to decrease depth to the same value as before downsampling):
            W_upsample = self.get_variable_weight_decay(scope + "/W_upsample",
                        shape=[1, 1, input_depth, output_depth], # ([filter_height, filter_width, in_depth, out_depth])
                        initializer=tf.contrib.layers.xavier_initializer(),
                        loss_category="decoder_wd_losses")
            main_branch = tf.nn.conv2d(main_branch, W_upsample, strides=[1, 1, 1, 1],
                        padding="VALID") # NOTE! no bias terms
            # # # batch norm:
            main_branch = tf.contrib.slim.batch_norm(main_branch)
            # NOTE! no ReLU here

            # # max unpooling:
            main_branch = max_unpool(main_branch, pooling_indices, output_shape)

        main_branch = tf.cast(main_branch, tf.float32)


        # convolution branch:
        conv_branch = x

        # # 1x1 projection:
        W_proj = self.get_variable_weight_decay(scope + "/W_proj",
                    shape=[1, 1, input_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="decoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_proj, strides=[1, 1, 1, 1],
                    padding="VALID") # NOTE! no bias terms
        # # # batch norm and ReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        conv_branch = tf.nn.relu(conv_branch)

        # # conv:
        if upsampling:
            # deconvolution:
            W_conv = self.get_variable_weight_decay(scope + "/W_conv",
                        shape=[3, 3, internal_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                        initializer=tf.contrib.layers.xavier_initializer(),
                        loss_category="decoder_wd_losses")
            b_conv = self.get_variable_weight_decay(scope + "/b_conv", shape=[internal_depth], # ([out_depth]], one bias weight per out depth layer),
                        initializer=tf.constant_initializer(0),
                        loss_category="decoder_wd_losses")
            main_branch_shape = main_branch.get_shape().as_list()
            output_shape = tf.convert_to_tensor([main_branch_shape[0],
                        main_branch_shape[1], main_branch_shape[2], internal_depth])
            conv_branch = tf.nn.conv2d_transpose(conv_branch, W_conv, output_shape=output_shape,
                        strides=[1, 2, 2, 1], padding="SAME") + b_conv
        else:
            W_conv = self.get_variable_weight_decay(scope + "/W_conv",
                        shape=[3, 3, internal_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                        initializer=tf.contrib.layers.xavier_initializer(),
                        loss_category="decoder_wd_losses")
            b_conv = self.get_variable_weight_decay(scope + "/b_conv", shape=[internal_depth], # ([out_depth])
                        initializer=tf.constant_initializer(0),
                        loss_category="decoder_wd_losses")
            conv_branch = tf.nn.conv2d(conv_branch, W_conv, strides=[1, 1, 1, 1],
                        padding="SAME") + b_conv
        # # # batch norm and ReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        conv_branch = tf.nn.relu(conv_branch)

        # # 1x1 expansion:
        W_exp = self.get_variable_weight_decay(scope + "/W_exp",
                    shape=[1, 1, internal_depth, output_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="decoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_exp, strides=[1, 1, 1, 1],
                    padding="VALID") # NOTE! no bias terms
        # # # batch norm:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        # NOTE! no ReLU here

        # NOTE! no regularizer


        # add the branches:
        merged = conv_branch + main_branch

        # apply ReLU:
        output = tf.nn.relu(merged)

        return output

    def get_variable_weight_decay(self, name, shape, initializer, loss_category, dtype=tf.float32):
        variable = tf.get_variable(name, shape=shape, dtype=dtype,
                    initializer=initializer)

        # add a variable weight decay loss:
        weight_decay = self.wd*tf.nn.l2_loss(variable)
        tf.add_to_collection(loss_category, weight_decay)
        return variable


