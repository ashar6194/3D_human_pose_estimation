from keras import backend as K
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import SeparableConv2D
from keras.layers import BatchNormalization, Conv2DTranspose, Conv1D
from keras.layers import GlobalMaxPooling2D
from keras.layers import TimeDistributed
from keras.layers import ZeroPadding2D
from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D
from keras.layers import GlobalMaxPooling1D

from keras.layers import multiply
from keras.layers import concatenate
from keras.layers import add


def conv(x, filters, size, strides=(1, 1), padding='same', name=None):
    x = Conv2D(filters, size, strides=strides, padding=padding,
            use_bias=False, name=name)(x)
    return x


def conv_pool(x, filters, size, strides=(1, 1), padding='valid', name=None):
  x = Conv2D(filters, size, strides=strides, padding=padding,
             use_bias=False, name=name, kernel_initializer='glorot_normal')(x)
  x = Activation('relu', name=name)(x)
  x = MaxPooling2D(strides, padding=padding)(x)
  return x


def conv_bn(x, filters, size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        conv_name = name + '_conv'
    else:
        conv_name = None

    x = conv(x, filters, size, strides, padding, conv_name)
    x = BatchNormalization(axis=-1, scale=False, name=name)(x)
    return x


def conv_bn_act(x, filters, size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        conv_name = name + '_conv'
        bn_name = name + '_bn'
    else:
        conv_name = None
        bn_name = None

    x = conv(x, filters, size, strides, padding, conv_name)
    x = BatchNormalization(axis=-1, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def act_conv_bn(x, filters, size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        conv_name = name + '_conv'
        act_name = name + '_act'
    else:
        conv_name = None
        act_name = None

    x = Activation('relu', name=act_name)(x)
    x = conv(x, filters, size, strides, padding, conv_name)
    x = BatchNormalization(axis=-1, scale=False, name=name)(x)
    return x


def separable_act_conv_bn(x, filters, size, strides=(1, 1), padding='same',
        name=None):
    if name is not None:
        conv_name = name + '_conv'
        act_name = name + '_act'
    else:
        conv_name = None
        act_name = None

    x = Activation('relu', name=act_name)(x)
    x = SeparableConv2D(filters, size, strides=strides, padding=padding,
            use_bias=False, name=conv_name)(x)
    x = BatchNormalization(axis=-1, scale=False, name=name)(x)
    return x


def act_conv(x, filters, size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        act_name = name + '_act'
    else:
        act_name = None

    x = Activation('relu', name=act_name)(x)
    x = conv(x, filters, size, strides, padding, name)
    return x


def conv_act(x, filters, size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        act_name = name + '_act'
    else:
        act_name = None

    x = conv(x, filters, size, strides, padding, name)
    x = Activation('relu', name=act_name)(x)
    return x


def act_channel_softmax(x, name=None):
    x = Activation(channel_softmax_2d(), name=name)(x)
    return x


def lin_interpolation_2d(inp, dim):

    num_rows, num_cols, num_filters = K.int_shape(inp)[1:]
    conv = SeparableConv2D(num_filters, (num_rows, num_cols), use_bias=False)
    x = conv(inp)

    w = conv.get_weights()
    w[0].fill(0)
    w[1].fill(0)
    linspace = linspace_2d(num_rows, num_cols, dim=dim)

    for i in range(num_filters):
        w[0][:,:, i, 0] = linspace[:,:]
        w[1][0, 0, i, i] = 1.

    conv.set_weights(w)
    conv.trainable = False

    x = Lambda(lambda x: K.squeeze(x, axis=1))(x)
    x = Lambda(lambda x: K.squeeze(x, axis=1))(x)
    x = Lambda(lambda x: K.expand_dims(x, axis=-1))(x)

    return x


def max_min_pooling(x, strides=(2, 2), padding='same', name=None):
    if 'max_min_pool_cnt' not in globals():
        global max_min_pool_cnt
        max_min_pool_cnt = 0

    if name is None:
        name = 'MaxMinPooling2D_%d' % max_min_pool_cnt
        max_min_pool_cnt += 1

    def _max_plus_min(x):
        x1 = MaxPooling2D(strides, padding=padding)(x)
        x2 = MaxPooling2D(strides, padding=padding)(-x)
        return x1 - x2

    return Lambda(_max_plus_min, name=name)(x)


def kronecker_prod(h, f, name='Kronecker_prod'):
    """ # Inputs: inp[0] (heatmaps) and inp[1] (visual features)
    """
    inp = [h, f]
    def _combine_heatmaps_visual(inp):
        hm = inp[0]
        x = inp[1]
        nj = K.int_shape(hm)[-1]
        nf = K.int_shape(x)[-1]
        hm = K.expand_dims(hm, axis=-1)
        hm = K.tile(hm, (1, 1, 1, 1, 1, nf))
        x = K.expand_dims(x, axis=-2)
        x = K.tile(x, (1, 1, 1, 1, nj, 1))
        x = hm * x
        x = K.sum(x, axis=(2, 3))
        return x
    return Lambda(_combine_heatmaps_visual, name=name)(inp)


def global_max_min_pooling(x, name=None):
    if 'global_max_min_pool_cnt' not in globals():
        global global_max_min_pool_cnt
        global_max_min_pool_cnt = 0

    if name is None:
        name = 'GlobalMaxMinPooling2D_%d' % global_max_min_pool_cnt
        global_max_min_pool_cnt += 1

    def _global_max_plus_min(x):
        x1 = GlobalMaxPooling2D()(x)
        x2 = GlobalMaxPooling2D()(-x)
        return x1 - x2

    return Lambda(_global_max_plus_min, name=name)(x)


def conv_traj(x, filters, size=1, strides=(1, 1), padding='valid', name=None):
    x = Conv1D(filters, size, padding=padding, use_bias=True,
               name=name)(x)
    return x


def conv_bn_traj(x, filters, size, strides=(1, 1), padding='valid', name=None):
    if name is not None:
        conv_name = name + '_conv'
    else:
        conv_name = None

    x = conv_traj(x, filters, size, strides, padding, conv_name)
    x = BatchNormalization(axis=-1, scale=False, name=name)(x)
    return x


def conv_bn_act_traj(x, filters, size, strides=(1, 1), padding='valid', name=None):
    if name is not None:
        conv_name = name + '_conv'
        bn_name = name + '_bn'
    else:
        conv_name = None
        bn_name = None

    x = conv_traj(x, filters, size, strides, padding, conv_name)
    x = BatchNormalization(axis=-1, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def deconv_traj(x, filters, size=1, strides=1, padding='valid', name=None):
    x = Conv2DTranspose(filters, kernel_size=(1, size), strides=strides, padding=padding, use_bias=True,
               name=name)(x)
    return x


def deconv_bn(x, filters, size, strides=(1, 1), padding='valid', name=None):
    if name is not None:
        conv_name = name + '_conv'
    else:
        conv_name = None

    x = deconv_traj(x, filters, size, strides, padding, conv_name)
    x = BatchNormalization(axis=-1, scale=False, name=name)(x)
    return x


def deconv_bn_act_traj(x, filters, size, strides=(1, 1), padding='valid', name=None):
    if name is not None:
        conv_name = name + '_conv'
        bn_name = name + '_bn'
    else:
        conv_name = None
        bn_name = None

    x = deconv_traj(x, filters, size, strides, padding, conv_name)
    x = BatchNormalization(axis=-1, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x