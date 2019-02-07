import glob
import pickle
import numpy as np
import cv2
import os

from scipy import io as sio
from transformations import rotation_matrix
from ubc_args import args


class generate_pc():
  def __init__(self):
    self.instance_list = sorted(glob.glob('%s*/groundtruth.mat' % args.root_dir))
    self.test_gt_list = sorted(glob.glob('%s*/gt_poses.pkl' % args.test_dir))
    self.test_pred_list = sorted(glob.glob('%s*/pred_%s_%s.pkl' % (args.test_dir, args.model_name, args.cam_type)))
    self.pose_list = sorted(glob.glob('%s*/gt_poses.pkl' % args.root_dir))
    self.parse_ubc_hard()

  def transform_mapper(self, mapper):
    fmap = mapper['mapper']
    fmap = np.squeeze(np.reshape(fmap, (-1, 1)))
    xmp, ymp = [], []

    for idx in range(len(fmap)):
      xmp.append(fmap[idx][0])
      ymp.append(fmap[idx][1])

    xmap, ymap = np.array(xmp).reshape((424, 512)), np.array(ymp).reshape((424, 512))
    xmap = np.flip(xmap, axis=-1)

    return xmap, ymap

  def dp_2_pc(self, depth_img, xmap, ymap, trans, rot):

    def convert_to_zdepth(img):
      return ((img[:, :, 1] * (750 / 255)) + 50) * 1.03

    xaxis, yaxis, zaxis = [1, 0, 0], [0, 1, 0], [0, 0, 1]
    cloud = np.zeros((424, 512, 3))
    dp_im = convert_to_zdepth(depth_img)

    cloud[:, :, 2] = dp_im[:, :]
    cloud[:, :, 0] = xmap * cloud[:, :, 2]
    cloud[:, :, 1] = ymap * cloud[:, :, 2]

    cloud = np.reshape(cloud, (-1, 3))

    rotyx = 180.0 - rot[0] if (rot[0] > 0) else (-rot[0])

    rotyy = rot[1]

    if (trans[0] * trans[2]) > 0:
      if rotyy < 0:
        rotyy = -rotyy
      else:
        rotyy = -180 + rotyy
    else:
      if rotyy < 0:
        rotyy = -180 + rotyy
      else:
        rotyy = -rotyy

    rot_mat = np.dot(rotation_matrix(rotyy, yaxis), rotation_matrix(rotyx, xaxis))
    cloud = np.dot(cloud, rot_mat[:3,:3])
    return cloud

  def parse_ubc_hard(self):
    mapper = '/home/mcao/Documents/human_pose_estimation/ubc_matlab/metadata/mapper.mat'
    mapp = sio.loadmat(mapper)

    for array_idx, (pose_array, instance) in enumerate(zip(self.pose_list, self.instance_list)):
      mat_gt_file = sio.loadmat(instance)
      cams = mat_gt_file['cameras'][0][0][0][0][0][0][0]

      pl_name = pose_array.split('/')[:-1]
      pl_name_im = '/'.join(pl_name) + '/images/depthRender/Cam1'

      pkl_array = pickle.load(open(pose_array, 'rb'))

      for idx in range(1001):
        translation = np.squeeze(cams[idx][0][0][0])
        rotation = np.squeeze(cams[idx][0][0][1])
        pose = pkl_array[idx, ]

        # self.publish_pose(pose / 100.0, array_idx, idx, data_mode='GT')

        im_name = os.path.join(pl_name_im, 'mayaProject.%06d.png' % (idx + 1))
        img = cv2.imread(im_name)
        xmap, ymap = self.transform_mapper(mapp)
        pts = self.dp_2_pc(img, xmap, ymap, translation, rotation)
        print img.shape, pts.shape


def main():
  generate_pc()


if __name__ == '__main__':
  main()