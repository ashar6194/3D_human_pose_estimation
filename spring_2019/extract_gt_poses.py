# This file is responsible for visualizing GT poses, depth maps and point clouds using numpy
# import rospy
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import tensorflow as tf
from scipy import io
from ubc_args import args
import glob
import time
from sklearn.cluster import KMeans


root_dir = args.root_dir
#
# joint_names = ['HeadPGX','HipsPGX', 'LeftArmPGX', 'LeftFingerBasePGX', 'LeftFootPGX', 'LeftForeArmPGX', 'LeftHandPGX','LeftInHandIndexPGX',
#                'LeftInHandThumbPGX', 'LeftLegPGX', 'LeftShoulderPGX', 'LeftToeBasePGX', 'LeftUpLegPGX','Neck1PGX', 'NeckPGX','RightArmPGX',
#                'RightFingerBasePGX','RightFootPGX', 'RightForeArmPGX', 'RightHandPGX', 'RightInHandIndexPGX', 'RightInHandThumbPGX',
#                'RightLegPGX', 'RightShoulderPGX', 'RightToeBasePGX', 'RightUpLegPGX', 'Spine1PGX', 'SpinePGX']
#
# idx_list = [0, 13, 23, 26, 27, 25, 22, 17, 24, 9, 4, 11, 18, 19, 16, 5, 6, 3]
# joint_name_list = [joint_names[i] for i in idx_list]


# def visualize_pose(pose_array):
#   fig = plt.figure(1)
#   ax = fig.add_subplot(111, projection='3d')
#   ax.scatter(pose_array[:, 0], pose_array[:, 1], pose_array[:, 2])

def clustering_pipeline(args):
  root_dir = args.root_dir
  instance_list = glob.glob('%s*/groundtruth.mat' % root_dir)
  pose_array = []

  for instance in tqdm(sorted(instance_list)):
    vid_idx = instance.split('/')[-2]
    pose_set = pickle.load(open('%s%s/gt_poses.pkl' % (root_dir, vid_idx), 'rb'))
    pose_array.append(pose_set)

  pose_array = np.array(pose_array).reshape((-1, 18, 3)).reshape((-1, 54))
  print pose_array.shape

  kmeans = KMeans(n_clusters=100, random_state=2, verbose=1, n_init=3)
  kmeans.fit(pose_array)

  pose_centroids = kmeans.cluster_centers_.reshape((-1, 18, 3))

  print pose_centroids.shape
  pickle.dump(pose_centroids, open('pose_centroids.pkl', 'wb'))


def find_train_stats(args):
  root_dir = args.root_dir
  instance_list = glob.glob('%s*/groundtruth.mat' % root_dir)
  pose_array = []
  data_stats = {}

  for instance in tqdm(sorted(instance_list)):
    vid_idx = instance.split('/')[-2]
    pose_set = pickle.load(open('%s%s/gt_poses.pkl' % (root_dir, vid_idx), 'rb'))
    pose_array.append(pose_set)

  pose_array = np.array(pose_array).reshape((-1, 18, 3)).reshape((-1, 54))
  data_stats['mean'] = np.mean(pose_array, axis=0)
  data_stats['std'] = np.std(pose_array, axis=0)

  pickle.dump(data_stats, open('datastats.pkl', 'wb'))

  print pose_array.shape



def main():
  instance_list = glob.glob('%s*/groundtruth.mat' % root_dir)
  flag_instance = False
  useful_jnt_idx = pickle.load(open('lookup.pkl', 'rb'))

  for instance in tqdm(sorted(instance_list)):
    mat_file = io.loadmat(instance)
    aa = time.time()
    vid_idx = instance.split('/')[-2]
    # fig = plt.figure()
    pose_list = []

    for idx in range(1001):
      # Create a dictionary of all existing joint names
      # if not flag_instance:
      #   print 'Hello!'
      #   flag_instance = True
      #   idx_dict = {}
      #   type_dict = str(mat_file['joints'][0][0].dtype).strip().split('), (')
      #   for limb_idx, limb_loc in enumerate(type_dict):
      #     j_name = limb_loc.strip().split(',')[-2][1:-1]
      #     idx_dict[j_name] = limb_idx
      #   useful_jnt_idx = [idx_dict[name] for name in joint_name_list]
      #   pickle.dump(useful_jnt_idx, open('lookup.pkl', 'wb'))

      joints = np.array([mat_file['joints'][0][idx][0][0][jnt_idx][0][12:15] for jnt_idx in useful_jnt_idx])
      pose_list.append(joints)

    pickle.dump(np.array(pose_list), open('%s%s/gt_poses.pkl' % (root_dir, vid_idx), 'wb'))
    # print time.time() - aa


if __name__ == '__main__':
  # main()
  # clustering_pipeline(args)
  find_train_stats(args)