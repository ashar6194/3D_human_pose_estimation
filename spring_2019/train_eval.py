import keras
import math
import os
import h5py
import glob
import numpy as np
import datetime

from keras import losses
from ubc_args import args
from model_set import build_ddp_basic, compile_network, build_ddp_vgg, build_minivgg_basic
from tdg import DataGenerator
from keras.models import load_model
from eval_pipeline import infer_outputs, eval_results

from keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint
from multiprocessing import cpu_count


def step_decay(epoch):
  initial_lrate = 0.001
  drop = 0.5
  epochs_drop = 1.0
  lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
  return lrate


if __name__ == '__main__':

  ckpt_dir = '/media/mcao/Miguel/UBC_hard/' + 'keras_models/'
  logs_dir = '/media/mcao/Miguel/UBC_hard/' + 'keras_logs/'

  if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

  if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

  inp_shape = (args.input_size, args.input_size, 1)

  img_train_list = sorted(glob.glob('%s*/images/depthRender/Cam1/*.png' % args.root_dir))
  img_test_list = sorted(glob.glob('%s*/images/depthRender/Cam1/*.png' % args.test_dir))
  train_dg = DataGenerator(img_train_list, batch_size=args.batch_size)
  test_dg = DataGenerator(img_test_list, batch_size=args.batch_size)

  if args.model_name == 'mini_vgg':
    print 'USing Model = %s' % args.model_name
    model = build_minivgg_basic(inp_shape, num_classes=100)

  elif args.model_name == 'mini_alex':
    model = build_ddp_basic(inp_shape, num_classes=100)

  # else:
  #   model = build_ddp_vgg(inp_shape, num_classes=100)
  # model.compile(loss=losses.mean_absolute_error, optimizer='Adam', metrics=['accuracy'])
  compile_network(model)

  filepath = ckpt_dir + 'weights_%03d.h5' % args.num_epochs
  checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                               save_best_only=True, period=5)
  tensorboard = TensorBoard(log_dir=logs_dir, batch_size=args.batch_size)
  lrate = LearningRateScheduler(step_decay)
  callbacks_list = [checkpoint, tensorboard, lrate]
  model.fit_generator(generator=train_dg, epochs=args.num_epochs, verbose=1, validation_data=test_dg,
                      use_multiprocessing=True, workers=cpu_count(), validation_steps=100)

  model_name = ckpt_dir + 'ddp_%s_ep50_cam1_%s.h5' % (args.model_name, datetime.datetime.now().strftime("%m_%d"))
  model.save(model_name)

  infer_outputs(args, model, args.test_dir)
  eval_results(args, args.test_dir)
  infer_outputs(args, model, args.root_dir)
  eval_results(args, args.root_dir)


  # h5py_file = h5py.File(model_name, 'w')
  # weight = model.get_weights()
  #
  # for i in range(len(weight)):
  #   h5py_file.create_dataset('weight' + str(i), data=weight[i])
  # h5py_file.close()
