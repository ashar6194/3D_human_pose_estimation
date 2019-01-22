import keras
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

  ckpt_dir = args.root_dir + 'keras_models/'

  inp_shape = (100, 100, 3)

  img_list = sorted(glob.glob('%s*/images/depthRender/Cam1/*.png' % args.root_dir))
  tdg = DataGenerator(img_list, flag_data='flow_kron', batch_size=1)

  model = build_ddp_basic(inp_shape, num_classes=54)
  model.compile(loss=losses.mean_absolute_error, optimizer='sgd', metrics=['accuracy'])
  # compile_network(model)
  model.fit_generator(generator=tdg, epochs=1, verbose=1)

  # use_multiprocessing=True, workers=4,

  model_name = ckpt_dir + 'model_final{}.hdf5'.format(datetime.datetime.now().strftime("%Y_%m_%d"))
  h5py_file = h5py.File(model_name, 'w')
  weight = model.get_weights()

  for i in range(len(weight)):
    h5py_file.create_dataset('weight' + str(i), data=weight[i])
  h5py_file.close()
