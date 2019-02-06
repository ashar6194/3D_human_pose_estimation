import h5py
import hdf5storage
import numpy as np
import cv2

from sys import getsizeof
from tqdm import tqdm

base_path = '/data/ITOP Dataset/itop_mats/'
itopset = h5py.File('/data/ITOP Dataset/ITOP_side_test_depth_map.h5', 'r+')
itop_pose = h5py.File('/data/ITOP Dataset/ITOP_side_test_labels.h5', 'r+')
keypoints = itop_pose['real_world_coordinates']

dp_img = itopset['data']
# keypoints = itop_pose['real_world_coordinates']
# keypoints  = np.array(keypoints).astype(np.float32)
# hdf5storage.write(keypoints, path='/poses', filename=base_path+'gt_poses.mat', matlab_compatible=True)

dp_img = np.array(dp_img).astype(np.float32)
dp_img[dp_img > 3.3] = 8

dp_img_arr = []

# rescaled_img = np.interp(dp_img, (dp_img.min(), dp_img.max()), (0, +1))

for i in tqdm(range(dp_img.shape[0])):
  img = dp_img[i, ]
  # img= np.resize(img, (100, 100))
  # img = np.repeat(img, 3, axis=-1)

  dst = np.zeros(shape=(5, 2))
  img = cv2.resize(img, (100, 100))
  img = cv2.normalize(img, dst, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
  # print np.min(img), np.max(img), np.mean(img)
  # print img.shape

  # cv2.imshow('im_name', img)
  # cv2.waitKey(1)

  dp_img_arr.append(img)

print getsizeof(dp_img_arr)

hdf5storage.write(np.array(dp_img_arr), path='/depth_image', filename=base_path+'all_dp_img.mat', matlab_compatible=True)


