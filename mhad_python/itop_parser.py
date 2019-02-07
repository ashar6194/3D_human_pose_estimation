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
keypoints = itop_pose['real_world_coordinates']
keypoints = np.array(keypoints).astype(np.float32).reshape((-1, 45))
keypoints = np.transpose(keypoints, (1, 0))
keypoints_mean = np.expand_dims(np.mean(keypoints, axis=-1), axis=-1)
keypoints_std = np.expand_dims(np.std(keypoints, axis=-1), axis=-1)

# norm_keypoints = (keypoints - keypoints_mean) / keypoints_std

print keypoints.shape

hdf5storage.write(keypoints, path='/joints_gt', filename=base_path+'gt_poses.mat', matlab_compatible=True)
hdf5storage.write(keypoints_mean, path='/joints_mean', filename=base_path+'gt_poses.mat', matlab_compatible=True)
hdf5storage.write(keypoints_std, path='/joints_std', filename=base_path+'gt_poses.mat', matlab_compatible=True)

dp_img = np.array(dp_img).astype(np.float32)
dp_img[dp_img > 4.5] = 8

dp_img_arr = []

# rescaled_img = np.interp(dp_img, (dp_img.min(), dp_img.max()), (0, +1))
#
# for i in tqdm(range(dp_img.shape[0])):
#   img = dp_img[i, ]
#   # img= np.resize(img, (100, 100))
#   # img = np.repeat(img, 3, axis=-1)
#
#   dst = np.zeros(shape=(5, 2))
#   img = cv2.resize(img, (100, 100))
#   img = cv2.normalize(img, dst, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#   # print np.min(img), np.max(img), np.mean(img)
#   # print img.shape
#
#   cv2.imshow('im_name', img)
#   cv2.waitKey(1)
#
#   dp_img_arr.append(img)
#
# # print getsizeof(dp_img_arr)
#
# dp_img_arr = np.array(dp_img_arr).transpose((1, 2, 0))
#
# hdf5storage.write(dp_img_arr, path='/data', filename=base_path+'all_dp_img.mat', matlab_compatible=True)
#

