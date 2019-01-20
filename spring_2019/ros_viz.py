import numpy as np
import glob
import rospy
import pickle

import cv2

from ubc_args import args

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Quaternion, Pose, Point, Vector3
from std_msgs.msg import Header, ColorRGBA

# from tqdm import tqdm


class pose_viz():

  def __init__(self):
    rospy.init_node('pose_visualizer')
    self.pose_list = sorted(glob.glob('%s*/gt_poses.pkl' % args.root_dir))
    print self.pose_list
    self.marker_publisher = rospy.Publisher('visualization_marker', Marker, queue_size=1)
    self.marker_id = 1

    print 'hey!!'
    self.main()

  def publish_pose(self, pose):
    points = Marker()
    points.header.frame_id = "/map"
    points.type = points.POINTS
    points.action = points.ADD
    points.id = self.marker_id
    self.marker_id = 1
    points.scale = (0.1, 0.1)
    # points.scale.y = 0.1
    points.color.a = 1.0
    points.color.r = 0.0
    points.color.g = 0.0
    points.color.b = 1.0

    # points.orientation = 1.0

    for i in range(18):
      point = Point(pose[i, 0], pose[i, 1], pose[i, 2])
      points.points.append(point)

    self.marker_publisher.publish(points)

  def main(self):
    while not rospy.is_shutdown():
      for pose_array in self.pose_list:
        pkl_array = pickle.load(open(pose_array, 'rb'))
        # print pkl_array.shape
        for idx in range(1001):
          pose = pkl_array[idx, ]
          self.publish_pose(pose/100.0)
          rospy.sleep(0.4)

      # break


if __name__ == '__main__':
  # try:
  pose_viz()
  # except:
  #   rospy.loginfo("Closing the Pose Visualization Node")
