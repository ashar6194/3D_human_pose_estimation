import argparse
import os
import time

colors = {
  'UBC_easy': [
    [255, 106, 0],
    [255, 0, 0],  #Upper Chest
    [255, 178, 127],
    [255, 127, 127],
    [182, 255, 0],
    [218, 255, 127],
    [255, 216, 0],
    [255, 233, 127],
    [0, 148, 255],
    [72, 0, 255],
    [48, 48, 48],
    [76, 255, 0],
    [0, 255, 33],
    [0, 255, 255],
    [0, 255, 144],
    [178, 0, 255],
    [127, 116, 63],
    [127, 63, 63],
    [127, 201, 255],
    [127, 255, 255],  #Left Thigh
    [165, 255, 127],
    [127, 255, 197],
    [214, 127, 255],
    [161, 127, 255],
    [107, 63, 127],
    [63, 73, 127],
    [63, 127, 127],
    [109, 127, 63],
    [255, 127, 237],
    [127, 63, 118],
    [0, 74, 127],
    [255, 0, 110],
    [0, 127, 70],
    [127, 0, 0],
    [33, 0, 127],
    [127, 0, 55],
    [38, 127, 0],
    [127, 51, 0],
    [64, 64, 64],
    [73, 73, 73],
    [0, 0, 0],        #Background
    [191, 168, 247],
    [192, 192, 192],
    [127, 63, 63],
    [127, 116, 63]
    ],

'UBC_MHAD': [
    [255, 106, 0],
    [255, 0, 0],
    [255, 178, 127],
    [255, 127, 127],
    [182, 255, 0],
    [218, 255, 127],
    [255, 216, 0],
    [255, 233, 127],
    [0, 148, 255],
    [72, 0, 255],
    [48, 48, 48],
    [76, 255, 0],
    [0, 255, 33],
    [0, 255, 255],
    [0, 255, 144],
    [178, 0, 255],
    [127, 116, 63],
    [127, 63, 63],
    [0, 0, 0]
    ],

'forced_UBC_easy': [
    [255, 106, 0],
    [255, 0, 0],  #Upper Chest
    [255, 178, 127],
    [255, 127, 127],
    [182, 255, 0],
    [218, 255, 127],
    [255, 216, 0],
    [255, 233, 127],
    [0, 148, 255],
    [72, 0, 255],
    [48, 48, 48],
    [76, 255, 0],
    [0, 255, 33],
    [0, 255, 255],
    [0, 255, 144],
    [178, 0, 255],
    [127, 116, 63],
    [127, 63, 63],
    [0, 0, 0]
    ],

  'MHAD': [
    [255, 106, 0],
    [255, 0, 0],
    [255, 178, 127],
    [255, 127, 127],
    [182, 255, 0],
    [218, 255, 127],
    [255, 216, 0],
    [255, 233, 127],
    [0, 148, 255],
    [72, 0, 255],
    [48, 48, 48],
    [76, 255, 0],
    [0, 255, 33],
    [0, 255, 255],
    [0, 255, 144],
    [178, 0, 255],
    [127, 116, 63],
    [127, 63, 63],
    [127, 201, 255],
    [127, 255, 255],
    [165, 255, 127],
    [127, 255, 197],
    [214, 127, 255],
    [161, 127, 255],
    [107, 63, 127],
    [63, 73, 127],
    [63, 127, 127],
    [109, 127, 63],
    [255, 127, 237],
    [127, 63, 118],
    [0, 74, 127],
    [255, 0, 110],
    [0, 127, 70],
    [127, 0, 0],
    [33, 0, 127],
    [0, 0, 0]        #Background
  ]
}

def get_str_file_name(working_dataset):
  str_file_name = 'frequencies_forced_UBC.txt'
  if working_dataset == 'MHAD':
      str_file_name = 'frequencies_MHAD.txt'
  elif working_dataset == 'UBC_easy':
      str_file_name = 'frequencies_UBC.txt'
  elif working_dataset == 'forced_UBC_easy':
      # str_file_name = 'frequencies_forced_UBC.txt'
      str_file_name = 'frequencies_forced_UBC_hard.txt'

  return str_file_name


# To make testing configuration easy
def get_directories(working_dataset):
  test_logs = './logs/logs_MHAD/'
  main_dir = '/media/mcao/Miguel/MHAD/Kinect/'
  ckpt_dir = '/media/mcao/Miguel/MHAD/ckpts/ckpts_MHAD/'
  main_output_dir = 'TestResults_correct_db'

  if working_dataset == 'MHAD':
    test_logs = './logs/logs_MHAD/test/sample_75PC'
    main_dir = '/media/mcao/Miguel/MHAD_fitted_plane/Kinect/'
    ckpt_dir = '/media/mcao/Miguel/MHAD/ckpts/ckpts_MHAD/sample_75PC'
    main_output_dir = 'Results_correct_fitted_plane'

  elif working_dataset == 'UBC_easy':
    test_logs = './logs/logs_UBC/original_12K/test'
    ckpt_dir = '/media/mcao/Miguel/MHAD/ckpts/ckpts_UBC/original_12K/'
    main_dir = '/media/mcao/Miguel/input/UBC_easy/train/'
    main_output_dir = '/media/mcao/Miguel/input/UBC_easy/original_12k/Results/'

  elif working_dataset == 'forced_UBC_easy':
    # test_logs = './logs/logs_UBC/forced_12K/test'
    # ckpt_dir = '/media/mcao/Miguel/MHAD/ckpts/ckpts_UBC/forced_12K/'
    # main_dir = '/media/mcao/Miguel/input/UBC_easy/train/'
    # main_output_dir = '/media/mcao/Miguel/input/UBC_easy/forced_12K/Results/'

    test_logs = './logs/logs_UBC/forced_UBC_hard/test'
    ckpt_dir = '/media/mcao/Miguel/MHAD/ckpts/ckpts_UBC/forced_UBC_hard/'
    main_dir = '/media/mcao/Miguel/UBC_hard/train/'
    main_output_dir = '/media/mcao/Miguel/UBC_hard/net_Results/'

  if working_dataset == 'UBC_MHAD':
    test_logs = './logs/logs_UBC_MHAD/forced_12K/test'
    main_dir = '/media/mcao/Miguel/MHAD/Kinect/'
    ckpt_dir = '/media/mcao/Miguel/MHAD/ckpts/ckpts_UBC/forced_12K/'
    main_output_dir = '/media/mcao/Miguel/MHAD/Kinect/forced_12K/Results/'

  return test_logs, ckpt_dir, main_dir, main_output_dir


def get_directories_training(working_dataset):
  log_directory = './logs/logs_MHAD/train/sample_75PC'
  ckpt_directory = '/media/mcao/Miguel/MHAD/ckpts/ckpts_MHAD/'

  if working_dataset == 'MHAD':
    log_directory = './logs/logs_MHAD/train/sample_75PC'
    ckpt_directory = '/media/mcao/Miguel/MHAD/ckpts/ckpts_MHAD/sample_75PC/'

  elif working_dataset == 'UBC_easy':
    log_directory = './logs/logs_UBC/original_12K/train'
    ckpt_directory = '/media/mcao/Miguel/MHAD/ckpts/ckpts_UBC/original_12K/'

  elif working_dataset == 'forced_UBC_easy':
    # log_directory = './logs/logs_UBC/forced_12K/train'
    # ckpt_directory = '/media/mcao/Miguel/MHAD/ckpts/ckpts_UBC/forced_12K/'

    log_directory = './logs/logs_UBC/forced_UBC_hard/train'
    ckpt_directory = '/media/mcao/Miguel/MHAD/ckpts/ckpts_UBC/forced_UBC_hard/'

  return log_directory, ckpt_directory


def get_csv_filename(working_dataset):

  filename = './csv/training_ubc_10k_sample.csv'
  if working_dataset == 'MHAD':
    filename = './csv/checkmhad.csv'
    # filename = './csv/training_mhad_fitted_planev2.csv'
  if working_dataset == 'UBC_easy':

    filename = './csv/training_ubc_10k_original.csv'
  if working_dataset == 'forced_UBC_easy':
    # filename = './csv/training_ubc_10k_forced.csv'
    filename = './csv/ubc_hard.csv'

  return filename


# # All the three dataset names
# 'MHAD'
# 'UBC_easy'
# 'forced_UBC_easy'
# 'UBC_MHAD'

autoencoder = 'segnet'
working_dataset = 'forced_UBC_easy'

gpu_memory_fraction = 0.75
strided = False
set = 'a'



