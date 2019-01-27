import numpy as np
import glob
import cv2
import pickle

from keras.models import load_model
from ubc_args import args


def infer_outputs(args, model):

  datastats = pickle.load(open('datastats.pkl', 'rb'))
  train_mean = datastats['mean']
  train_std = datastats['std']
  pkl_array = pickle.load(open('pose_centroids.pkl', 'rb'))

  img_folder_list = sorted(glob.glob('%s*/images/depthRender/*' % args.test_dir))
  for id_folder in img_folder_list:
    img_name_list = sorted(glob.glob('%s/*.png' % id_folder))
    print 'Hey!'

    pose_list = []
    for id_name in img_name_list:
      print id_name

      img = cv2.imread(id_name)
      # aa = time.time()
      img = cv2.resize(img, (args.input_size, args.input_size))
      dst = np.zeros(shape=(5, 2))
      img = cv2.normalize(img, dst, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
      X = np.expand_dims(img, axis=0)
      pose_weights = model.predict(X)

      gt_bases = np.reshape(pkl_array, (100, -1))
      gt_bases = (gt_bases - train_mean) / train_std

      pred_rep = np.repeat(np.reshape(pose_weights, (-1, 100, 1)), gt_bases.shape[1], axis=2)
      final_pred_comb = pred_rep * gt_bases
      final_pred = np.sum(final_pred_comb, axis=1)
      pose_list.append(final_pred)

    pose_npy = np.array(pose_list)

    name_parse = id_folder.split('/')
    vid_idx = name_parse[-4]
    pred_file = '%s%s/pred_poses.pkl' % (args.test_dir, vid_idx)
    pickle.dump(pose_npy, open(pred_file, 'wb'))


if __name__ == '__main__':
  ckpt_dir = '/media/mcao/Miguel/UBC_hard/' + 'keras_models/'
  model_name = ckpt_dir + 'model_cam1_2019_01_26.h5'
  ddp_model = load_model(model_name)
  infer_outputs(args, ddp_model)
