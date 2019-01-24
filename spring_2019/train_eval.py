import keras
import os
import h5py
import glob
import numpy as np
import datetime

from keras import losses
from ubc_args import args
from model_set import build_ddp_basic, compile_network
from tdg import DataGenerator
from keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint


if __name__ == '__main__':

  ckpt_dir = '/media/mcao/Miguel/UBC_hard/' + 'keras_models/'
  if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

  inp_shape = (args.input_size, args.input_size, 3)

  img_train_list = sorted(glob.glob('%s*/images/depthRender/*/*.png' % args.root_dir))
  img_test_list = sorted(glob.glob('%s*/images/depthRender/*/*.png' % args.test_dir))
  train_dg = DataGenerator(img_train_list, batch_size=args.batch_size)
  test_dg = DataGenerator(img_test_list, batch_size=args.batch_size)

  model = build_ddp_basic(inp_shape, num_classes=54)
  # model.compile(loss=losses.mean_absolute_error, optimizer='Adam', metrics=['accuracy'])
  compile_network(model)
  model.fit_generator(generator=train_dg, epochs=3, verbose=1, validation_data=test_dg)

  # use_multiprocessing=True, workers=4,

  model_name = ckpt_dir + 'model_final{}.hdf5'.format(datetime.datetime.now().strftime("%Y_%m_%d"))
  h5py_file = h5py.File(model_name, 'w')
  weight = model.get_weights()

  for i in range(len(weight)):
    h5py_file.create_dataset('weight' + str(i), data=weight[i])
  h5py_file.close()
