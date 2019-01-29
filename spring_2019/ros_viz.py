import numpy as np
import glob
import rospy
import pickle
import os

import cv2
# from tqdm import tqdm
from ubc_args import args

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Quaternion, Pose, Point, Vector3
from std_msgs.msg import Header, ColorRGBA
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class pose_viz():

  def __init__(self):
    rospy.init_node('pose_visualizer')
    self.pose_list = sorted(glob.glob('%s*/gt_poses.pkl' % args.root_dir))

    self.test_gt_list = sorted(glob.glob('%s*/gt_poses.pkl' % args.test_dir))
    self.test_pred_list = sorted(glob.glob('%s*/pred_%s_%s.pkl' % (args.test_dir, args.model_name, args.cam_type)))
    print self.test_pred_list

    # print self.pose_list
    self.bridge = CvBridge()
    self.marker_publisher = rospy.Publisher('visualization_marker', Marker, queue_size=50)

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
        points=[Point(points_pair[0, 0], points_pair[0, 1] , points_pair[0, 2]), Point(points_pair[1, 0], points_pair[1, 1] , points_pair[1, 2])]
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

  def visualize_train_data_only(self):
    while not rospy.is_shutdown():
      for array_idx, pose_array in enumerate(self.pose_list):
        pl_name = pose_array.split('/')[:-1]
        pl_name_im = '/'.join(pl_name) + '/images/depthRender/Cam1'

        pkl_array = pickle.load(open(pose_array, 'rb'))

        for idx in range(1001):
          pose = pkl_array[idx, ]
          self.publish_pose(pose/100.0, array_idx, idx, data_mode='GT')

          im_name = os.path.join(pl_name_im, 'mayaProject.%06d.png' % (idx+1))
          img = cv2.imread(im_name)

          try:
            self.im_publisher.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))
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

          # self.publish_pose(gt_pose/100.0, array_idx, idx, data_mode='GT')
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