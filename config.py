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

'UBC_medium': [
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

'UBC_hard': [
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

  'MHAD_UBC': [
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
  ],

  'UBC_interpolated': [
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

  elif working_dataset == 'UBC_easy' or working_dataset == 'UBC_medium' \
      or working_dataset == 'UBC_hard':
      str_file_name = 'frequencies_UBC.txt'
  elif working_dataset == 'forced_UBC_easy':
      # str_file_name = 'frequencies_forced_UBC.txt'
      str_file_name = 'frequencies_forced_UBC_hard.txt'
  elif working_dataset == 'UBC_interpolated':
      str_file_name = 'frequencies_UBC_interpolated.txt'
  return str_file_name


# To make testing configuration easy
def get_directories(working_dataset):
  test_logs = '/media/mcao/Miguel/human_pose_estimation/logs/logs_MHAD/'
  main_dir = '/media/mcao/Miguel/MHAD/Kinect/'
  ckpt_dir = '/media/mcao/Miguel/MHAD/ckpts/ckpts_MHAD/'
  main_output_dir = 'TestResults_correct_db'

  if working_dataset == 'MHAD':
    test_logs = '/media/mcao/Miguel/human_pose_estimation/logs/logs_MHAD/test/sample_75PC'
    main_dir = '/media/mcao/Miguel/MHAD_fitted_plane/Kinect/'
    ckpt_dir = '/media/mcao/Miguel/MHAD/ckpts/ckpts_MHAD/sample_75PC'
    main_output_dir = 'Results_75PC'

  elif working_dataset == 'UBC_easy':
    test_logs = '/media/mcao/Miguel/human_pose_estimation/logs/logs_UBC/original_easy/test'
    ckpt_dir = '/media/mcao/Miguel/MHAD/ckpts/ckpts_UBC/original_easy/'
    main_dir = '/media/mcao/Miguel/input/UBC_easy/train/'
    main_output_dir = '/media/mcao/Miguel/input/UBC_easy/original_12k/Results/'

  elif working_dataset == 'UBC_hard':
    test_logs = '/media/mcao/Miguel/human_pose_estimation/logs/logs_UBC/original_hard/test/'
    ckpt_dir = '/media/mcao/Miguel/MHAD/ckpts/ckpts_UBC/original_hard/train/'
    main_dir = '/media/mcao/Miguel/UBC_hard/test/'
    main_output_dir = '/media/mcao/Miguel/UBC_hard/original_hard/net_Results/test/'

  elif working_dataset == 'forced_UBC_easy':
    # test_logs = './logs/logs_UBC/forced_12K/test'
    # ckpt_dir = '/media/mcao/Miguel/MHAD/ckpts/ckpts_UBC/forced_12K/'
    # main_dir = '/media/mcao/Miguel/input/UBC_easy/train/'
    # main_output_dir = '/media/mcao/Miguel/input/UBC_easy/forced_12K/Results/'

    test_logs = '/media/mcao/Miguel/human_pose_estimation/logs/logs_UBC/forced_UBC_hard_v2/test'
    ckpt_dir = '/media/mcao/Miguel/MHAD/ckpts/ckpts_UBC/forced_UBC_hard_v2/'
    main_dir = '/media/mcao/Miguel/UBC_hard/train/'
    main_output_dir = '/media/mcao/Miguel/UBC_hard/net_Results_v2/train/'

  if working_dataset == 'UBC_MHAD':
    test_logs = '/media/mcao/Miguel/human_pose_estimation/logs/logs_UBC_MHAD/UBC_hard/test'

    # main_dir = '/media/mcao/Miguel/MHAD/Kinect/'
    main_dir = '/media/mcao/Miguel/MHAD_fitted_plane/Kinect/'

    # ckpt_dir = '/media/mcao/Miguel/MHAD/ckpts/ckpts_UBC/forced_12K/'
    ckpt_dir = '/media/mcao/Miguel/MHAD/ckpts/ckpts_UBC/forced_UBC_hard_v2/'
    main_output_dir = '/media/mcao/Miguel/MHAD_fitted_plane/Kinect/forced_UBC_hard_v2/Results/'

  if working_dataset == 'MHAD_UBC':
    test_logs = '/media/mcao/Miguel/human_pose_estimation/logs/logs_UBC/MHAD_UBC/test/'
    ckpt_dir = '/media/mcao/Miguel/MHAD/ckpts/ckpts_MHAD/sample_75PC'
    main_dir = '/media/mcao/Miguel/UBC_hard/valid/'
    main_output_dir = '/media/mcao/Miguel/MHAD_UBC/Segnet_Results/valid/'

  elif working_dataset == 'UBC_interpolated':

    # To infer UBC-Hard
    test_logs = '/media/mcao/Miguel/human_pose_estimation/logs/logs_UBC/UBC_interpolated/test'
    ckpt_dir = '/media/mcao/Miguel/MHAD/ckpts/ckpts_UBC/UBC_interpolated/'
    main_dir = '/media/mcao/Miguel/UBC_hard/test/'
    main_output_dir = '/media/mcao/Miguel/UBC_hard/interpolated_Results/test/'

    # To Infer MHAD
    # test_logs = '/media/mcao/Miguel/human_pose_estimation/logs/logs_UBC/UBC_interpolated/test'
    # ckpt_dir = '/media/mcao/Miguel/MHAD/ckpts/ckpts_UBC/UBC_interpolated/'
    # main_dir = '/media/mcao/Miguel/MHAD_fitted_plane/Kinect/'
    # main_output_dir = '/media/mcao/Miguel/MHAD_fitted_plane/Kinect/UBC_interpolated/Results/'

  return test_logs, ckpt_dir, main_dir, main_output_dir


def get_directories_training(working_dataset):
  log_directory = './media/mcao/Miguel/human_pose_estimation/logs/logs_MHAD/train/sample_75PC'
  ckpt_directory = '/media/mcao/Miguel/MHAD/ckpts/ckpts_MHAD/'

  if working_dataset == 'MHAD':
    log_directory = '/media/mcao/Miguel/human_pose_estimation/logs/logs_MHAD/train/sample_75PC'
    ckpt_directory = '/media/mcao/Miguel/MHAD/ckpts/ckpts_MHAD/sample_75PC/'

  elif working_dataset == 'UBC_easy':
    log_directory = '/media/mcao/Miguel/human_pose_estimation/logs/logs_UBC/original_easy/train'
    ckpt_directory = '/media/mcao/Miguel/MHAD/ckpts/ckpts_UBC/original_easy/train/'

  elif working_dataset == 'UBC_medium':
    log_directory = '/media/mcao/Miguel/human_pose_estimation/logs/logs_UBC/original_medium/train'
    ckpt_directory = '/media/mcao/Miguel/MHAD/ckpts/ckpts_UBC/original_medium/train/'

  elif working_dataset == 'UBC_hard':
    log_directory = '/media/mcao/Miguel/human_pose_estimation/logs/logs_UBC/original_hard/train'
    ckpt_directory = '/media/mcao/Miguel/MHAD/ckpts/ckpts_UBC/original_hard/train/'

  elif working_dataset == 'forced_UBC_easy':
    # log_directory = './logs/logs_UBC/forced_12K/train'
    # ckpt_directory = '/media/mcao/Miguel/MHAD/ckpts/ckpts_UBC/forced_12K/'

    log_directory = '/media/mcao/Miguel/human_pose_estimation/logs/logs_UBC/forced_UBC_hard_v2/train'
    ckpt_directory = '/media/mcao/Miguel/MHAD/ckpts/ckpts_UBC/forced_UBC_hard_v2/'

  elif working_dataset == 'UBC_interpolated':
    log_directory = '/media/mcao/Miguel/human_pose_estimation/logs/logs_UBC/UBC_interpolated/train'
    ckpt_directory = '/media/mcao/Miguel/MHAD/ckpts/ckpts_UBC/UBC_interpolated/'

  return log_directory, ckpt_directory


def get_csv_filename(working_dataset):

  filename = './csv/training_ubc_10k_sample.csv'
  if working_dataset == 'MHAD':
    filename = './csv/checkmhad.csv'

    # filename = './csv/training_mhad_fitted_planev2.csv'
  if working_dataset == 'UBC_easy':
    filename = './csv/original_ubc_easy.csv'

  if working_dataset == 'UBC_medium':
    filename = './csv/original_ubc_medium.csv'

  if working_dataset == 'UBC_hard':
    filename = './csv/original_ubc_hard.csv'

  if working_dataset == 'forced_UBC_easy':
    # filename = './csv/training_ubc_10k_forced.csv'
    filename = './csv/ubc_hard.csv'

  if working_dataset == 'UBC_interpolated':
    filename = './csv/interpolated_ubc_hard.csv'

  return filename


# # All the three dataset names
# 'MHAD'
# 'UBC_easy'
# 'forced_UBC_easy'
# 'UBC_MHAD'
# 'MHAD_UBC'
# 'UBC_interpolated'

autoencoder = 'segnet'
working_dataset = 'UBC_hard'

gpu_memory_fraction = 0.85
strided = False
set = 'a'