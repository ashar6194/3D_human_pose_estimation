from keras.models import Model, Sequential
import numpy as np
from keras import optimizers
from keras.activations import softmax
import pickle
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


def custom_L1_loss(y_true, y_pred):
  sigma_sq = 0.8
  alpha = 0.01

  pkl_array = pickle.load(open('pose_centroids.pkl', 'rb'))
  alpha = 0.2
  gt_bases = K.cast(K.reshape(pkl_array, (100, -1)), 'float32')
  pred_rep = K.repeat_elements(K.reshape(y_pred, (-1, 100, 1)), gt_bases.shape[1], axis=2)
  final_pred = pred_rep * gt_bases
  final_pred = K.sum(final_pred, axis=1)

  res_term = K.abs(y_true - final_pred)
  res_loss = Lambda(lambda x: K.switch(x > (1/sigma_sq), x - (0.5/sigma_sq), 0.5 * sigma_sq * x * x))(res_term)
  res_loss = K.sum(res_loss)
  reg_term = K.sum(K.abs(y_pred))
  return (1 - alpha) * res_loss + alpha * reg_term


  # res_1, res_2 = res_term, res_term
  #
  # res_1[res_1 < (1/sigma_sq)] = 0
  # res_1 -= (0.5/sigma_sq)
  # res_2[res_2 >= (1 / sigma_sq)] = 0
  # res_2 = 0.5 * sigma_sq * res_2 * res_2
  # res_loss = res_1 + res_2



def compile_network(model):
  # optim_adam = optimizers.Adam(lr=0.001)
  model.compile(loss=custom_L1_loss, optimizer='sgd')


if __name__== '__main__':
  print keras.__version__
  custom_L1_loss(1, 2)






