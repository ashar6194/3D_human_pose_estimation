import numpy as np
import glob
import rospy
import pickle
import os
import math
import cv2
# from tqdm import tqdm
from ubc_args import args
from scipy import io as sio
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Quaternion, Pose, Point, Vector3
from std_msgs.msg import Header, ColorRGBA
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge, CvBridgeError
from transformations import rotation_matrix


class pose_viz():

  def __init__(self):
    rospy.init_node('pose_visualizer')
    self.pose_list = sorted(glob.glob('%s*/gt_poses.pkl' % args.root_dir))
    self.instance_list = sorted(glob.glob('%s*/groundtruth.mat' % args.root_dir))

    self.test_gt_list = sorted(glob.glob('%s*/gt_poses.pkl' % args.test_dir))
    self.test_pred_list = sorted(glob.glob('%s*/pred_%s_%s.pkl' % (args.test_dir, args.model_name, args.cam_type)))
    print self.test_pred_list

    # print self.pose_list
    self.bridge = CvBridge()
    self.marker_publisher = rospy.Publisher('visualization_marker', Marker, queue_size=50)
    self.pub_cloud = rospy.Publisher("/pcl_msg", PointCloud2, queue_size=100)

    # self.gt_marker_publisher = rospy.Publisher('visualization_marker', Marker, queue_size=50)
    # self.pred_marker_publisher = rospy.Publisher('visualization_marker', Marker, queue_size=50)

    self.im_publisher = rospy.Publisher('dp_image', Image, queue_size=50)
    self.marker_id = 1
    self.main()

  def publish_individual_marker(self, idx, npy_point, data_mode):
    if data_mode == 'GT':
      marker = Marker(
        type=Marker.SPHERE,
        id=idx+1,
        lifetime=rospy.Duration(0),
        pose=Pose(Point(npy_point[0], npy_point[1], npy_point[2]), Quaternion(0, 0, 0, 1)),
        scale=Vector3(0.05, 0.05, 0.05),
        header=Header(frame_id='/map'),
        color=ColorRGBA(0, 1, 0.0, 1)
      )
    else:
      marker = Marker(
        type=Marker.SPHERE,
        id=idx+40,
        lifetime=rospy.Duration(0),
        pose=Pose(Point(npy_point[0], npy_point[1], npy_point[2]), Quaternion(0, 0, 0, 1)),
        scale=Vector3(0.05, 0.05, 0.05),
        header=Header(frame_id='/map'),
        color=ColorRGBA(0, 0, 1.0, 1)
      )

    self.marker_publisher.publish(marker)

  def publish_individual_line(self, idx, points_pair, data_mode):
    if data_mode == 'GT':
      # print 'Hey'
      marker4 = Marker(
        type=Marker.LINE_STRIP,
        id=idx+20,
        lifetime=rospy.Duration(0),
        pose=Pose(Point(0,0,0), Quaternion(0, 0, 0, 1)),
        scale=Vector3(0.025, 0.025, 0.025),
        header=Header(frame_id='/map'),
        color=ColorRGBA(1.0, 0.0, 1.0, 1),
        points=[Point(points_pair[0, 0], points_pair[0, 1], points_pair[0, 2]), Point(points_pair[1, 0], points_pair[1, 1] , points_pair[1, 2])]
      )

    else:
      marker4 = Marker(
        type=Marker.LINE_STRIP,
        id=idx+60,
        lifetime=rospy.Duration(0),
        pose=Pose(Point(0,0,0), Quaternion(0, 0, 0, 1)),
        scale=Vector3(0.025, 0.025, 0.025),
        header=Header(frame_id='/map'),
        color=ColorRGBA(1.0, 0.0, 0.0, 1),
        points=[Point(points_pair[0, 0], points_pair[0, 1], points_pair[0, 2]), Point(points_pair[1, 0], points_pair[1, 1] , points_pair[1, 2])]
      )
    self.marker_publisher.publish(marker4)

  def publish_pose(self, pose, array_idx, idx, data_mode='GT'):
    for i in range(18):
      self.publish_individual_marker(i, pose[i, ], data_mode)

    pairlist = [[0, 1], [1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11],
                [3, 12], [12, 13], [13, 14], [3, 15], [15, 16], [16, 17]]
    # idx_list = [0, 13, 23, 26, 27, 25, 22, 17, 24, 9, 4, 11, 18, 19, 16, 5, 6, 3]

    for pair_idx, pair in enumerate(pairlist):
      pp1 = pose[pair[0]]
      pp2 = pose[pair[1]]

      npa = np.array([pp1, pp2])
      # print npa.shape
      self.publish_individual_line(pair_idx, npa, data_mode)

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

    offset = 50

    def convert_to_zdepth(img):
      return ((img[:, :, 0] * (750 / 255)) + offset) * 1.03

    alpha_thresh = 75

    alphac = depth_img[:, :, 3]

    flattened_alpha = np.ravel(alphac.T)
    id = np.arange(flattened_alpha.shape[0])
    cond_idx = id[flattened_alpha > alpha_thresh]

    xaxis, yaxis, zaxis = [1, 0, 0], [0, 1, 0], [0, 0, 1]
    cloud = np.zeros((424, 512, 3))
    dp_im = convert_to_zdepth(depth_img)

    cloud[:, :, 2] = dp_im[:, :]
    cloud[:, :, 0] = xmap * cloud[:, :, 2]
    cloud[:, :, 1] = ymap * cloud[:, :, 2]

    cloud = np.reshape(cloud, (-1, 3))
    # cloud = cloud[(cloud[:, 2] > offset + 0.2)]

    if (rot[0] > 0):
      rotyx = 180.0 - rot[0]
    else:
      rotyx = (-rot[0])

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

    rot_mat = np.dot(rotation_matrix(0, zaxis), np.dot(rotation_matrix(math.radians(rotyy), yaxis), rotation_matrix(math.radians(rotyx), xaxis)))

    cloud = np.dot(cloud, rot_mat[:3, :3].T)
    cloud += trans
    # cloud = cloud[cond_idx, :]

    return cloud

  def xyzToPcl(self, msg_pcl, pts):
    flag_dense = int(np.isfinite(pts).all())
    msg_pcl.data = np.asarray(pts, np.float32).tostring()
    msg_pcl.is_dense = flag_dense
    msg_pcl.height = 1
    msg_pcl.width = pts.shape[0]
    msg_pcl.fields = [
      PointField('x', 0, PointField.FLOAT32, 1),
      PointField('y', 4, PointField.FLOAT32, 1),
      PointField('z', 8, PointField.FLOAT32, 1)]
    msg_pcl.is_bigendian = False
    msg_pcl.point_step = 12
    msg_pcl.row_step = 12 * pts.shape[0]
    msg_pcl.header.stamp = rospy.Time.now()
    msg_pcl.header.frame_id = 'map'
    return msg_pcl

  def visualize_train_data_only(self):
    while not rospy.is_shutdown():
      mapper = '/home/mcao/Documents/human_pose_estimation/ubc_matlab/metadata/mapper.mat'
      mapp = sio.loadmat(mapper)
      xmap, ymap = self.transform_mapper(mapp)

      for array_idx, (pose_array, instance) in enumerate(zip(self.pose_list, self.instance_list)):
        mat_gt_file = sio.loadmat(instance)
        cams = mat_gt_file['cameras'][0][0][0][0][0][0][0]
        pl_name = pose_array.split('/')[:-1]
        pl_name_im = '/'.join(pl_name) + '/images/depthRender/Cam1'

        pkl_array = pickle.load(open(pose_array, 'rb'))

        for idx in range(1001):
          translation = np.squeeze(cams[idx][0][0][0])
          rotation = np.squeeze(cams[idx][0][0][1])

          # if not idx:
            # print translation, rotation
            # break
          pose = pkl_array[idx, ]
          self.publish_pose(pose/100.0, array_idx, idx, data_mode='GT')

          im_name = os.path.join(pl_name_im, 'mayaProject.%06d.png' % (idx+1))
          img = cv2.imread(im_name, cv2.IMREAD_UNCHANGED)

          # print img.shape

          pts = self.dp_2_pc(img, xmap, ymap, translation, rotation)
          print pts.shape
          pcl = PointCloud2()

          pcl_msg = self.xyzToPcl(pcl, pts/100.0)
          self.pub_cloud.publish(pcl_msg)

          try:
            self.im_publisher.publish(self.bridge.cv2_to_imgmsg(img[:, :, :3], "bgr8"))
            # print 'Hey!'
          except CvBridgeError as e:
            print(e)
          rospy.sleep(0.5)


  def visualize_train_test(self):
    while not rospy.is_shutdown():
      for array_idx, (gt_array, pred_array) in enumerate(zip(self.test_gt_list, self.test_pred_list)):
        pl_name = gt_array.split('/')[:-1]
        pl_name_im = '/'.join(pl_name) + '/images/depthRender/Cam1'

        pkl_gt = pickle.load(open(gt_array, 'rb'))
        pkl_pred = pickle.load(open(pred_array, 'rb'))

        for idx in range(1001):
          gt_pose = pkl_gt[idx, ]
          pred_pose = pkl_pred[idx, ]

          # print gt_pose.shape

          self.publish_pose(gt_pose/100.0, array_idx, idx, data_mode='GT')
          self.publish_pose(pred_pose/100.0, array_idx, idx, data_mode='PRED')

          im_name = os.path.join(pl_name_im, 'mayaProject.%06d.png' % (idx+1))
          img = cv2.imread(im_name)

          try:
            self.im_publisher.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))
            # print 'Hey!'
          except CvBridgeError as e:
            print(e)
          rospy.sleep(1)

  def visualize_cluster_centroids(self):
    while not rospy.is_shutdown():
      pkl_array = pickle.load(open('pose_centroids.pkl', 'rb'))
      for idx in range(100):
        pose = pkl_array[idx, ]
        self.publish_pose(pose/100.0, 0, idx)
        rospy.sleep(0.5)

  def main(self):
    if args.show_td == 'train_data_only':
      self.visualize_train_data_only()
    elif args.show_td == 'train_test':
      self.visualize_train_test()
    else:
      self.visualize_cluster_centroids()


if __name__ == '__main__':
  pose_viz()