import numpy as np
import glob
import cv2
import pickle

from tqdm import tqdm
from keras.models import load_model
from ubc_args import args
from model_set import build_ddp_basic, compile_network, build_ddp_vgg, build_minivgg_basic


def eval_results(args, directory=args.test_dir):
  pred_cam_list = sorted(glob.glob('%s*/pred_%s_%s.pkl' % (directory, args.model_name, args.cam_type)))
  gt_pose_list = sorted(glob.glob('%s*/gt_poses.pkl' % directory))

  full_pred_list = []
  full_gt_list = []

  for pred_pose_file, gt_pose_file in zip(pred_cam_list, gt_pose_list):
    pred_array = pickle.load(open(pred_pose_file, 'rb'))
    gt_array = pickle.load(open(gt_pose_file, 'rb'))

    full_pred_list.append(pred_array)
    full_gt_list.append(gt_array)

  full_pred_npy = np.array(full_pred_list)
  full_gt_npy = np.array(full_gt_list)

  error = np.linalg.norm((full_gt_npy - full_pred_npy), axis=-1).reshape(-1, 18)

  print np.mean(error)
  print np.median(error)



def infer_outputs(args, model, directory=args.test_dir):

  datastats = pickle.load(open('datastats.pkl', 'rb'))
  train_mean = datastats['mean']
  train_std = datastats['std']
  pkl_array = pickle.load(open('pose_centroids.pkl', 'rb'))

  img_folder_list = sorted(glob.glob('%s*/images/depthRender/Cam1' % directory))
  for id_folder in tqdm(img_folder_list):
    img_name_list = sorted(glob.glob('%s/*.png' % id_folder))

    pose_list = []
    for id_name in img_name_list:
      # print id_name

      img = cv2.imread(id_name)
      img = cv2.resize(img, (args.input_size, args.input_size))
      dst = np.zeros(shape=(5, 2))
      img = cv2.normalize(img, dst, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
      img = np.expand_dims(np.mean(img, axis=-1), axis=-1)
      X = np.expand_dims(img, axis=0)
      pose_weights = model.predict(X)

      gt_bases = np.reshape(pkl_array, (100, -1))
      gt_bases = (gt_bases - train_mean) / train_std

      pred_rep = np.repeat(np.reshape(pose_weights, (-1, 100, 1)), gt_bases.shape[1], axis=2)
      final_pred_comb = pred_rep * gt_bases
      final_pred = np.sum(final_pred_comb, axis=1)
      pose_list.append(final_pred)

    pose_npy = np.squeeze(np.array(pose_list))
    pose_npy = (pose_npy * train_std) + train_mean
    pose_npy = pose_npy.reshape((-1, 18, 3))
    # print pose_npy.shape

    name_parse = id_folder.split('/')
    vid_idx = name_parse[-4]
    cam_idx = name_parse[-1]
    pred_file = '%s%s/preds_%s_%s.pkl' % (directory, vid_idx, args.model_name, cam_idx)
    # print pred_file

    pickle.dump(pose_npy, open(pred_file, 'wb'))


if __name__ == '__main__':
  eval_results(args)
  # ckpt_dir = '/media/mcao/Miguel/UBC_hard/' + 'keras_models/'
  # model_name = ckpt_dir + 'ddp_mini_vgg_cam1_2019_01_29.h5'
  # inp_shape = (args.input_size, args.input_size, 1)
  # ddp_model = build_minivgg_basic(inp_shape, num_classes=100)
  # ddp_model.load_weights(model_name)
  # infer_outputs(args, ddp_model)
