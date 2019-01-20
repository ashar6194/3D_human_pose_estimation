# This file is responsible for visualizing GT poses, depth maps and point clouds using numpy
# import rospy
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import tensorflow as tf
from scipy import io
import glob

root_dir = '/media/mcao/Miguel/UBC_hard/train/'

joint_names = ['HeadPGX','HipsPGX', 'LeftArmPGX', 'LeftFingerBasePGX', 'LeftFootPGX', 'LeftForeArmPGX', 'LeftHandPGX','LeftInHandIndexPGX',
               'LeftInHandThumbPGX', 'LeftLegPGX', 'LeftShoulderPGX', 'LeftToeBasePGX', 'LeftUpLegPGX','Neck1PGX', 'NeckPGX','RightArmPGX',
               'RightFingerBasePGX','RightFootPGX', 'RightForeArmPGX', 'RightHandPGX', 'RightInHandIndexPGX', 'RightInHandThumbPGX',
               'RightLegPGX', 'RightShoulderPGX', 'RightToeBasePGX', 'RightUpLegPGX', 'Spine1PGX', 'SpinePGX']

idx_list = [0, 13, 23, 26, 27, 25, 22, 17, 24, 9, 4, 11, 18, 19, 16, 5, 6, 3]
joint_name_list = [joint_names[i] for i in idx_list]


def visualize_pose(pose_array):
  fig = plt.figure(1)
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(pose_array[:, 0], pose_array[:, 1], pose_array[:, 2])


def main():
  instance_list = glob.glob('%s*/groundtruth.mat' % root_dir)
  flag_instance = False

  for inst_idx, instance in tqdm(enumerate(sorted(instance_list))):

    fig = plt.figure()
    pose_list = []

    for idx in range(1001):
      mat_file = io.loadmat(instance)
      # Create a dictionary of all existing joint names
      if not flag_instance:
        print 'Hello!'
        flag_instance = True
        idx_dict = {}
        type_dict = str(mat_file['joints'][inst_idx][0].dtype).strip().split('), (')
        for limb_idx, limb_loc in enumerate(type_dict):
          j_name = limb_loc.strip().split(',')[-2][1:-1]
          idx_dict[j_name] = limb_idx

        useful_jnt_idx = [idx_dict[name] for name in joint_name_list]

      joints = np.array([mat_file['joints'][0][idx][0][0][jnt_idx][0][12:15] for jnt_idx in useful_jnt_idx])
      pose_list.append(joints)

    pickle.dump(np.array(pose_list), open('%s%d/gt_poses.pkl' % (root_dir, (inst_idx+1)), 'wb'))


if __name__ == '__main__':
  main()
