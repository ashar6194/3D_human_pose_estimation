import keras

import os
import h5py
import glob
import numpy as np
import datetime

from keras import losses
from ubc_args import args
from model_set import build_ddp_basic, compile_network, build_ddp_vgg
from tdg import DataGenerator
from keras.models import load_model

from keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint
from multiprocessing import cpu_count


if __name__ == '__main__':

  ckpt_dir = '/media/mcao/Miguel/UBC_hard/' + 'keras_models/'
  logs_dir = '/media/mcao/Miguel/UBC_hard/' + 'keras_logs/'

  if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

  if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

  inp_shape = (args.input_size, args.input_size, 3)

  img_train_list = sorted(glob.glob('%s*/images/depthRender/*/*.png' % args.root_dir))
  img_test_list = sorted(glob.glob('%s*/images/depthRender/*/*.png' % args.test_dir))
  train_dg = DataGenerator(img_train_list, batch_size=args.batch_size)
  test_dg = DataGenerator(img_test_list, batch_size=args.batch_size)

  model = build_ddp_vgg(inp_shape, num_classes=100)
  # model.compile(loss=losses.mean_absolute_error, optimizer='Adam', metrics=['accuracy'])
  compile_network(model)

  filepath = ckpt_dir + 'weights_{epoch:02d}.h5'
  checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
  tensorboard = TensorBoard(log_dir=logs_dir, batch_size=args.batch_size)
  callbacks_list = [checkpoint, tensorboard]
  model.fit_generator(generator=train_dg, epochs=3, verbose=1, validation_data=test_dg,
                      use_multiprocessing=True, workers=cpu_count(), validation_steps=35)

  # use_multiprocessing=True, workers=4,

  model_name = ckpt_dir + 'model_cam1_{}.h5'.format(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))
  model.save(model_name)


  # h5py_file = h5py.File(model_name, 'w')
  # weight = model.get_weights()
  #
  # for i in range(len(weight)):
  #   h5py_file.create_dataset('weight' + str(i), data=weight[i])
  # h5py_file.close()
