# 3 CM- Height
# 17 CM- Lateral


from models import MiniAutoencoder, MiniUNet, MiniResSum, SegNetAutoencoder, MiniFusion, MiniFusionNet, MobileUnet
from newmodels import ENet_model
from sklearn.cluster import KMeans
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Quaternion, Pose, Point, Vector3
from std_msgs.msg import Header, ColorRGBA

import config
import os
import tensorflow as tf
import numpy as np
import math
import cv2
import rospy
import itertools

def colors_of_dataset(dataset_name):
  return config.colors[dataset_name]

def get_autoencoder(autoencoder_name, dataset_name, strided):
  n_labels = len(colors_of_dataset(dataset_name))
  autoencoders = {
    'mini': MiniAutoencoder,
    'mini_unet': MiniUNet,
    'mini_rsum': MiniResSum,
    'mini_munet': MobileUnet,
    'mini_fusion': MiniFusion,
    'mnfnet': MiniFusionNet,
    'enet': ENet_model,
    'segnet': SegNetAutoencoder
  }
  return autoencoders[autoencoder_name](n_labels, strided=strided)


def get_dataset(dataset_name, include_labels, subdir, kind):
  path = os.path.join('input/UBC_easy/TFrecords/%s' % subdir, dataset_name)
  data_binary_path = os.path.join(path, '%s.tfrecords' % kind)
  if include_labels:
    labels_binary_path = os.path.join(path, '%s_labels.tfrecords' % kind)
    return data_binary_path, labels_binary_path
  return data_binary_path


def get_training_set(dataset_name, include_labels=True):
  return get_dataset(dataset_name, include_labels, 'train')


def get_test_set(dataset_name, include_labels=False):
  return get_dataset(dataset_name, include_labels, 'test')


def restore_logs(logfile):
  if tf.gfile.Exists(logfile):
    tf.gfile.DeleteRecursively(logfile)
  tf.gfile.MakeDirs(logfile)

def minDistance(pt, pts_list):
      # print 'Size of pts list - ' + str(pts_list.shape)
      dist_2 = np.sum((pts_list - pt)**2, axis=1)
      return pts_list[np.argmin(dist_2), :]

def limitbyclosestpoints(pts_list, closest_pts, mindistrange=0.25):
      dist_mask = np.sqrt(np.sum((pts_list - closest_pts)**2, axis=1)) < mindistrange
      pts_list_masked = pts_list[dist_mask, :]
      # print np.count_nonzero(dist_mask)
      # print pts_list_masked.shape
      return pts_list_masked

def findMidPoint(npcloud):

  last_axis = npcloud.shape[1]
  # print 'last_axis = ', str(last_axis)

  origin = np.zeros(last_axis)
  center_init = np.array([[-1, 3], [1, 3]])

  kmeans = KMeans(n_clusters=2) #, init = center_init, n_init = 1)
          # # Fitting the input data
  kmeans = kmeans.fit(npcloud)
  # Centroid values
  centroids = kmeans.cluster_centers_
  labels_inferred = kmeans.labels_

  inferred_set = np.concatenate([npcloud, np.expand_dims(labels_inferred, axis=1)], axis=1)
  pts_list_a = inferred_set[inferred_set[:,last_axis] == 1]
  pts_list_a = pts_list_a[:,:last_axis]
  pts_list_b = inferred_set[inferred_set[:,last_axis] == 0]
  pts_list_b = pts_list_b[:,:last_axis]

  min_dist_a = minDistance(origin, pts_list_a)
  min_dist_b = minDistance(origin, pts_list_b)

  min_dist_center_a = minDistance(centroids[0, :], pts_list_a)
  min_dist_center_b = minDistance(centroids[1, :], pts_list_b)

  # Within a range of 25 cm
  # cp_set_a = limitbyclosestpoints(pts_list_a, min_dist_a)
  # cp_set_b = limitbyclosestpoints(pts_list_b, min_dist_b)

  # poi_a = np.mean(cp_set_a, axis=0)
  # poi_b = np.mean(cp_set_b, axis=0)

  return min_dist_a, min_dist_b, centroids[0, :], centroids[1, :]


def pointcloudXYZ(masked_depth_image, camera_info, ymin=-0.6, ymax=0.6, zmin=0, zmax=10000):
    proj_matrix_ravel =  np.asarray(camera_info.P)
    proj_matrix = np.reshape(proj_matrix_ravel, (3, 4))
    fx = proj_matrix[0, 0]
    fy = proj_matrix[1, 1]
    cx = proj_matrix[0, 2]
    cy = proj_matrix[1, 2]

    x, y = np.meshgrid(range(masked_depth_image.shape[1]), range(masked_depth_image.shape[0]))

    x1_cor = 0.001 * (x - cx) * masked_depth_image / fx
    y1_cor = 0.001 * (y - cy) * masked_depth_image / fy

    xr, yr, x1_corr, y1_corr, flattened_depth_image = np.ravel(x.T), np.ravel(y.T), np.ravel(x1_cor.T), np.ravel(y1_cor.T), np.ravel(masked_depth_image.T)
    idx = np.arange(flattened_depth_image.shape[0])
    cond_idx = idx[(zmin < flattened_depth_image) & (zmax > flattened_depth_image)]
    pts = np.array([x1_corr[cond_idx[:]], y1_corr[cond_idx[:]], 0.001 *flattened_depth_image[cond_idx[:]]]).T

    pts = pts[(pts[:, 1] < ymax)]
    pts = pts[(pts[:, 1] > ymin)]
    # print pts.shape, np.min(pts[:, 1]),  np.max(pts[:, 1])
    pts_2d = np.delete(pts, 1, 1)
    # print pts_2d.shape
    return pts, pts_2d

def xyzToPcl(msg_pcl, pts, flag_dense):
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
  msg_pcl.row_step = 12*pts.shape[0]
  msg_pcl.header.stamp = rospy.Time.now()
  return msg_pcl

def findMaxAreaComponents(binary_mask, num_regions=2):
    n = num_regions+1
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, 4, cv2.CV_32S)
    maxtwoarea = np.argsort(stats[:, 4])[-n:]
    newmask = binary_mask
    newmask[:] = 0
    for i in range(n-1):
      newmask[labels==maxtwoarea[i-n]] = 255

    return newmask

def histCrop(masked_depth_image):
    # First and last indices of non-zero pixels in positive - x direction
    itemidy = np.where(masked_depth_image.sum(axis=0) != 0)
    mid_point = (np.max(itemidy[0]) + np.min(itemidy[0])) / 2

    img_a, img_b = masked_depth_image[:,:mid_point+1],  masked_depth_image[:,mid_point+1:]
    # print img_a.shape, img_b.shape

    hist_a, __ = np.histogram(img_a.ravel(),600,[0,6000]);
    hist_b, __ = np.histogram(img_b.ravel(),600,[0,6000]);

    peakidx1 = np.argmax(hist_a[1:])
    peakidx2 = np.argmax(hist_b[1:])

    print peakidx1, peakidx2, (peakidx1 + peakidx2) / 200.0

def getRotMat(trans, theta):

  mat = np.array([[np.cos(theta), -np.sin(theta), trans[0]], 
      [np.sin(theta), np.cos(theta), trans[1]], 
      [0, 0, 1]
      ])
  return mat


def invertCameraFrame(marker1_x, marker2_x, marker1_y, marker2_y):
  theta = np.arctan2((marker2_y - marker1_y), (marker2_x - marker1_x)) if marker1_x < marker2_x else np.arctan2((marker1_y - marker2_y), (marker1_x - marker2_x))
  trans = [(marker2_x + marker1_x)/2, (marker2_y + marker1_y)/2, 1]

  gCM = getRotMat(trans, theta)
  gMC = np.linalg.inv(gCM)
  camera_origin = np.array([0, 0, 1])
  trans_obj = np.expand_dims(np.array(camera_origin), axis=1)
  pt_mp_frame = np.matmul(gMC, trans_obj)
  x, z = pt_mp_frame[0],  pt_mp_frame[1]
  return x, -z


def getTheta(marker1, marker2, is2D=True):

  if is2D:
    theta = -np.arctan2((marker2[1] - marker1[1]), (marker2[0] - marker1[0])) if marker1[0] < marker2[0] else -np.arctan2((marker1[1] - marker2[1]), (marker1[0] - marker2[0]))
    
  elif marker1[0] < marker2[0]:
    theta = np.zeros(3)
    theta[1] = -np.arctan2((marker2[2] - marker1[2]), (marker2[0] - marker1[0]))
    theta[0] = -np.arctan2((marker2[2] - marker1[2]), (marker2[1] - marker1[1]))
    theta[2] = -np.arctan2((marker2[1] - marker1[1]), (marker2[0] - marker1[0]))
  else:
    theta = np.zeros(3)
    theta[1] = -np.arctan2((marker1[2] - marker2[2]), (marker1[0] - marker2[0]))
    theta[0] = -np.arctan2((marker1[2] - marker2[2]), (marker1[1] - marker2[1]))
    theta[2] = -np.arctan2((marker1[1] - marker2[1]), (marker1[0] - marker2[0]))

  return theta


def cluster2D(img, mask):
  
  output = cv2.bitwise_and(img, img, mask = mask)
  # output = img[:,:,2] * mask
  print 'Output Dimensions'
  print output.shape
  print output.dtype
  # output = np.expand_dims(output, axis=2)
  # output = np.concatenate([output, np.concatenate([output, output], axis=2)], axis=2)

  whitepixels = np.fliplr((np.asarray(np.where(mask != 0)).T))
  med = np.mean(whitepixels, axis = 0)

  kmeans = KMeans(n_clusters=2)
  kmeans = kmeans.fit(whitepixels)

  centroids = kmeans.cluster_centers_
  labels_inferred = kmeans.labels_

  inferred_set = np.concatenate([whitepixels, np.expand_dims(labels_inferred, axis=1)], axis=1)

  pts_list_a = inferred_set[inferred_set[:,2] == 1]
  pts_list_a = pts_list_a[:,:2]
  pts_list_b = inferred_set[inferred_set[:,2] == 0]
  pts_list_b = pts_list_b[:,:2]

  min_dist_a = minDistance(centroids[0, :], pts_list_a)
  min_dist_b = minDistance(centroids[1, :], pts_list_b)

  center_mean = np.mean(centroids, axis=0)

  mid_min_dist = (min_dist_a + min_dist_b)/ 2

  # To Visiualize the location of inferred clusters
  withCircle = cv2.circle(output, (int(math.ceil(med[0])), int(math.ceil(med[1]))), 5, (0,255,0), 3)
  # Edge Points
  # withCircleA = cv2.circle(output, (int(math.ceil(min_dist_a[0])), int(math.ceil(min_dist_a[1]))), 5, (255,0,0), 3)
  # withCircleB = cv2.circle(output, (int(math.ceil(min_dist_b[0])), int(math.ceil(min_dist_b[1]))), 5, (255,0,0), 3)
  # Mean of Closest Points
  withCircleC = cv2.circle(output, (int(math.ceil(mid_min_dist[0])), int(math.ceil(mid_min_dist[1]))), 5, (255,255,0), 3)
  # Mean of CLuster Centers
  withCircleD = cv2.circle(output, (int(math.ceil(center_mean[0])), int(math.ceil(center_mean[1]))), 5, (0,165,255), 3)
  # # CLuster Centers
  withCircleE = cv2.circle(output, (int(math.ceil(centroids[0, 0])), int(math.ceil(centroids[0, 1]))), 5, (0,255,255), 3)
  withCircleF = cv2.circle(output, (int(math.ceil(centroids[1, 0])), int(math.ceil(centroids[1, 1]))), 5, (0,255,255), 3)

  print mid_min_dist - center_mean
  return mid_min_dist, mask, output

def generateVisualizationMarkers(mid_point, centroid_mid_point, cam_frame_id, markerA=[0], markerB=[0]):
  marker1 = Marker(
    type=Marker.SPHERE,
    id=0,
    lifetime=rospy.Duration(0),
    pose=Pose(Point(0, 0, 0), Quaternion(0, 0, 0, 1)),
    scale=Vector3(0.025, 0.025, 0.025),
    header=Header(frame_id='base_link'),
    color=ColorRGBA(0, 1, 0.0, 1))
  marker1.header.frame_id = cam_frame_id

    # Second end of MP
  marker2 = Marker(
    type=Marker.SPHERE,
    id=1,
    lifetime=rospy.Duration(0),
    pose=Pose(centroid_mid_point, Quaternion(0, 0, 0, 1)),
    scale=Vector3(0.025, 0.025, 0.025),
    header=Header(frame_id='base_link'),
    color=ColorRGBA(1.0, 0.0, 0.0, 1))
  marker2.header.frame_id = cam_frame_id

  # marker3 = Marker(
  #   type=Marker.SPHERE,
  #   id=2,
  #   lifetime=rospy.Duration(0),
  #   pose=Pose(Point(0, 0, 0), Quaternion(0, 0, 0, 1)),
  #   scale=Vector3(0.25, 0.25, 0.25),
  #   header=Header(frame_id='camera_link'),
  #   color=ColorRGBA(0.0, 0.0, 1.0, 1))
  # marker3.header.frame_id = cam_frame_id

  # marker4 = Marker(
  #   type=Marker.LINE_STRIP,
  #   id=3,
  #   lifetime=rospy.Duration(0),
  #   pose=Pose(Point(0, 0, 0), Quaternion(0, 0, 0, 1)),
  #   scale=Vector3(0.05, 0.05, 0.05),
  #   header=Header(frame_id='base_link'),
  #   color=ColorRGBA(1.0, 0.0, 1.0, 1),
  #   points=[Point(markerB[0], 0, markerB[1]), Point(markerA[0], 0,  markerA[1])])
  # marker4.header.frame_id = cam_frame_id

  markerArray = MarkerArray()
  markerArray.markers = [marker1, marker2]
  return markerArray

def findMidPointL2(npcloud, num_clusters=3):

  last_axis = npcloud.shape[1]
  # print 'last_axis = ', str(last_axis)

  origin = np.zeros(last_axis)
  center_init = np.array([[-1, 3], [1, 3]])

  kmeans = KMeans(n_clusters=num_clusters) #, init = center_init, n_init = 1)
          # # Fitting the input data
  kmeans = kmeans.fit(npcloud)
  # Centroid values
  centroids = kmeans.cluster_centers_
  labels_inferred = kmeans.labels_

  inferred_set = np.concatenate([npcloud, np.expand_dims(labels_inferred, axis=1)], axis=1)

  pts_list = []
  cpf = np.array([])

  for i in range(num_clusters):
    pts_list.append(inferred_set[inferred_set[:,last_axis] == i])
    pts_list[i] = pts_list[i][:,:last_axis]
    cps = np.expand_dims(minDistance(origin, pts_list[i]), axis=0)
    cpf = np.concatenate([cpf, cps], axis=0) if cpf.size else cps

  final_pts_list = []

  if len(pts_list) > 2:
    idxpair = findConstrainedClusters(cpf)
    final_pts_list.append(pts_list[idxpair[0]])
    final_pts_list.append(pts_list[idxpair[1]])
    # print idxpair
    # print len(final_pts_list)

  else:
    final_pts_list = pts_list[0:2]

  min_dist_a = minDistance(origin, final_pts_list[0])
  min_dist_b = minDistance(origin, final_pts_list[1])


  # Within a range of 25 cm
  # cp_set_a = limitbyclosestpoints(pts_list_a, min_dist_a)
  # cp_set_b = limitbyclosestpoints(pts_list_b, min_dist_b)

  # poi_a = np.mean(cp_set_a, axis=0)
  # poi_b = np.mean(cp_set_b, axis=0)

  return min_dist_a, min_dist_b, centroids[0, :], centroids[1, :]

def findConstrainedClusters(centroids):
  dist = 2.223
  pairlist = list(itertools.combinations(range(centroids.shape[0]), 2))

  # print pairlist
  it = 0
  dist_L2 = np.zeros(centroids.shape[0])
  for pair in pairlist:
    dist_L2[it] = np.sqrt(np.sum((centroids[pair[0], :] - centroids[pair[1], :])**2))
    if dist_L2[it] > 1.8 and dist_L2[it] < 2.70:
      print dist_L2[it]
      return pair
  return pair


