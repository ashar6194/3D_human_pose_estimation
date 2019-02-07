''' Furthest point sampling
Original author: Haoqiang Fan
Modified by Charles R. Qi
All Rights Reserved. 2017.
'''
from hdf5_util import *
from tqdm import  tqdm
import glob
import cPickle as pickle
import h5py
# import pickle
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
# from scipy.spatial import Delaunay
# from scipy.spatial import
import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sampling_module = tf.load_op_library(os.path.join(BASE_DIR, 'tf_sampling_so.so'))


def plot_basic_object(points, pose):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  S = ax.scatter(points[0, :], points[1, :], points[2, :], c='b', marker='o')
  P = ax.plot(pose[0, :], pose[1, :], pose[2, :], c='r', marker='x')
  plt.show()


def prob_sample(inp, inpr):
  return sampling_module.prob_sample(inp, inpr)


ops.NoGradient('ProbSample')


def gather_point(inp, idx):

  return sampling_module.gather_point(inp, idx)


@tf.RegisterGradient('GatherPoint')
def _gather_point_grad(op, out_g):
  inp = op.inputs[0]
  idx = op.inputs[1]
  return [sampling_module.gather_point_grad(inp, idx, out_g), None]


def farthest_point_sample(npoint, inp):
  '''
input:
  int32
  batch_size * ndataset * 3   float32
returns:
  batch_size * npoint         int32
  '''
  return sampling_module.farthest_point_sample(inp, npoint)


ops.NoGradient('FarthestPointSample')

if __name__ == '__main__':
  import numpy as np

  # np.random.seed(100)
  # triangles = np.random.rand(1, 5, 3, 3).astype('float32')

  mat_filelist = 'pc_filelist.txt'
  # input_dir = '/media/mcao/Miguel/MHAD_fitted_plane/Kinect/pointclouds/1/'
  ubc_dir = '/data/UBC_hard/train_pc/'
  ubc_GT_dir = '/data/UBC_hard/train_pcgt/'
  output_dir = '/data/UBC_hard/train_pc_2048/'
  tensor_filenames = sorted(glob.glob('%s*.mat' % ubc_dir))
  # pose_filenames = sorted(glob.glob('%s*.mat' % ubc_GT_dir))

  N = len(tensor_filenames) - 1
  sample_num = 2048
  mean_array, std_array = [], []
  min_array, max_array = [], []

  for i in tqdm(range(N)):
    out_file_nm = tensor_filenames[i].split('/')[-1].split('.')[0]

    if not os.path.exists('%s%s.pkl' % (output_dir, out_file_nm)):
      # mat = sio.loadmat(tensor_filenames[i])
      mat = h5py.File(tensor_filenames[i], 'r')
      # pose = h5py.File(pose_filenames[i], 'r')

      d = np.array(mat['pts'])
      # l = np.array(pose['jnt'])
      # p = mat['joints']
      # plot_basic_object(d)
      # print 'Dimensions for the mat file -- \n'
      d = np.expand_dims(d, 0).astype('float32')
      # l = np.expand_dims(l, 0).astype('float32')

      # print d.shape, l.shape

      with tf.device('/gpu:0'):
        hp_sample = tf.constant(d)
        # hp_labels = tf.constant(l)

        reduced_sample = gather_point(hp_sample, farthest_point_sample(sample_num, hp_sample))
        # reduced_labels = gather_point(hp_labels, farthest_point_sample(1024, hp_labels))

      with tf.Session('') as sess:
        points = sess.run(reduced_sample)
        points = np.squeeze(np.transpose(points, (0, 2, 1)))

        mean_npy = np.mean(points, axis=-1)
        std_npy = np.std(points, axis=-1)
        min_npy = np.min(points, axis=-1)
        max_npy = np.max(points, axis=-1)

        spcl = {}

        spcl['points'] = points
        spcl['min'] = min_npy
        spcl['max'] = max_npy
        spcl['mean'] = mean_npy
        spcl['std'] = std_npy

        # pose_viz = np.squeeze(np.transpose(l, (0, 2, 1)))
        # plot_basic_object(zxc/200.0, pose_viz/200.0)
        # plot_basic_object(pose_viz, 'r')
        a = 1
        pickle.dump(spcl, open('%s%s.pkl' % (output_dir, out_file_nm), 'wb'))
