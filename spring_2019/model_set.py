from keras.models import Model, Sequential

from keras import optimizers
from keras.activations import softmax

import keras

from sklearn.metrics import confusion_matrix

from layers import *


def build_ddp_basic(input_shape, num_classes=54):
  inp = Input(shape=input_shape)
  # x = conv_pool(inp, 64, (3, 3))
  x = conv_pool(inp, 64, (3, 3))
  x = conv_pool(inp, 128, (3, 3))
  # x = conv_pool(inp, 128, (3, 3))

  # x = conv_pool(x, 192, (5, 5))
  # x = conv_pool(x, 512, (3, 3))
  # x = conv_act(x, 1024, (2, 2))
  # x = conv_act(x, 2048, (2, 2))
  x = Flatten()(x)
  # x = Dense(1024)(x)

  x = Dense(256)(x)
  x = Dropout(rate=0.2)(x)
  x_out = Dense(num_classes)(x)
  model = Model(inputs=inp, outputs=x_out, name='Base Model')

  return model


def compile_network(model):
  # optim_adam = optimizers.Adam(lr=0.001)
  model.compile(loss='mean_squared_error', optimizers='sgd', metrics=['accuracy'])


if __name__== '__main__':
  print keras.__version__







