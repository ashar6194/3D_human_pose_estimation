from __future__ import absolute_import, division
from tqdm import *
from view_pc import plotxyz
import pickle
import numpy as np
from scipy import misc
# import h5py
from config import colors as color_map
import utils

pickle_fv_inferred = open("../ckpts/ckpts_mhad_python/features_inferred_s3.pickle", "wb")
pickle_gt_inferred = open("../ckpts/ckpts_mhad_python/ground_truth_inferred_s3.pickle", "wb")


def extract_features_raw(drc, config_file, subjects, actions, recordings):
    feature_dictionary = dict((str(k).decode(), []) for k in range(1, subjects + 1))
    gt_dictionary = dict((str(k).decode(), []) for k in range(1, subjects + 1))

    # Fetch Camera parameters
    fx1, fy1, cx1, cy1, fx2, fy2, cx2, cy2, h1, h2 = utils.get_camera_params(config_file)

    for sub in range(1, subjects + 1):

        # Declare Feature and ground truth lists
        feature, gt = np.array([]), np.array([])

        for act in range(1, actions + 1):
            for rec in range(1, recordings + 1):
                try:
                    skeleton_file = drc + 'Skeletons/skl_s%02d_a%02d_r%02d.mat' % (sub, act, rec)
                    im_mc, im_mc2, skel_jnt, center, stand = utils.get_skeleton_info(skeleton_file)

                    print '\nExtraction of features in subject ' + str(sub) + ' action ' + str(
                        act) + ' recording ' + str(rec)

                    for i in tqdm(range(min(im_mc.shape[1], im_mc2.shape[1]))):
                        try:
                            # Read Depth data from two kinects
                            d1 = misc.imread(
                                '%s/Kinect/Kin01/S%02d/A%02d/R%02d/kin_k01_s%02d_a%02d_r%02d_depth_%05d.pgm' % (
                                    drc, sub, act, rec, sub, act, rec, i))
                            d2 = misc.imread(
                                '%s/Kinect/Kin02/S%02d/A%02d/R%02d/kin_k02_s%02d_a%02d_r%02d_depth_%05d.pgm' % (
                                    drc, sub, act, rec, sub, act, rec, i))

                            # Apply inverse projection to fetch point clouds in cm
                            pts1, __, __ = utils.map_range_2_pc(d1, fx1, fy1, cx1, cy1, h1, center, stand, thresh=5)
                            pts2, __, __ = utils.map_range_2_pc(d2, fx2, fy2, cx2, cy2, h2, center, stand, thresh=5)

                            full_pc = np.hstack([pts1, pts2])
                            full_pc = full_pc[[1, 2, 0], :]
                            plotxyz(full_pc.T, color='b', hold=False)

                            skel = skel_jnt.T
                            jnt = np.reshape(skel[int(im_mc[2, i]) + 1, :], (35, 3)).T

                            plotxyz(jnt.T, color='r', hold=False)

                            # nearest neighbour
                            diff = full_pc.T - np.reshape(jnt, (3, 1, 35)).T
                            dfsq = np.sqrt(np.sum((diff ** 2), axis=2))
                            label_idx = np.argmin(dfsq, axis=0)
                            full_pcl = np.vstack([full_pc, np.reshape(label_idx, (1, label_idx.shape[0]))])

                            # store joint-wise features
                            feature_image = np.array([])

                            for joint in range(jnt.shape[1]):
                                ind_seg = full_pcl[3, :] == joint
                                pcl_joint = full_pcl[0:3, ind_seg]
                                pcl_joint[np.isnan(pcl_joint)] = 0

                                # extract moments
                                med_joint = np.median(pcl_joint, 1) if pcl_joint.size else np.zeros(3)
                                med_joint[np.isnan(med_joint)] = 0

                                std_joint = np.std(pcl_joint, 1) if pcl_joint.size else np.zeros(3)
                                std_joint[np.isnan(std_joint)] = 0

                                min_joint = np.min(pcl_joint, 1) if pcl_joint.size else np.zeros(3)
                                min_joint[np.isnan(min_joint)] = 0

                                max_joint = np.max(pcl_joint, 1) if pcl_joint.size else np.zeros(3)
                                max_joint[np.isnan(max_joint)] = 0

                                cov_joint = np.cov(pcl_joint) if pcl_joint.size else np.zeros((3, 3))
                                cov_joint[np.isnan(cov_joint)] = 0

                                eig_joint = np.linalg.eigvals(cov_joint)
                                eig_joint[np.isnan(eig_joint)] = 0

                                feature_joint = np.concatenate(
                                    [np.ravel(med_joint), np.ravel(std_joint), np.ravel(min_joint),
                                     np.ravel(max_joint), np.ravel(eig_joint), np.ravel(cov_joint),
                                     ], axis=0)

                                feature_image = np.concatenate([np.ravel(feature_image), np.ravel(feature_joint)]) \
                                    if feature_image.size else feature_joint

                            feature = np.vstack([feature, feature_image]) if feature.size else feature_image
                            # print feature.shape
                            gt = np.vstack([gt, np.ravel(jnt)]) if gt.size else np.ravel(jnt)

                        except IndexError:
                            print '\nNumber ' + str(i) + ' image is empty!'

                except IOError:
                    print '\nFile does not exist!'

        feature_dictionary[str(sub)].append(feature)
        gt_dictionary[str(sub)].append(gt)

    return feature_dictionary, gt_dictionary


def extract_features_labelled(drc, config_file, dir_labels, mhad_color_map, subjects, actions, recordings):

    # Fetch Camera parameters
    fx1, fy1, cx1, cy1, fx2, fy2, cx2, cy2, h1, h2 = utils.get_camera_params(config_file)

    for sub in range(1, subjects + 1):

        feature_dictionary = {str(sub).decode(): []}
        gt_dictionary = {str(sub).decode(): []}

        # Declare Feature and ground truth lists
        feature, gt = np.array([]), np.array([])

        for act in range(1, actions + 1):
            for rec in range(1, recordings + 1):
                try:
                    skeleton_file = drc + 'Skeletons/skl_s%02d_a%02d_r%02d.mat' % (sub, act, rec)
                    im_mc, im_mc2, skel_jnt, center, stand = utils.get_skeleton_info(skeleton_file)

                    print '\nExtraction of features in subject ' + str(sub) + ' action ' + str(
                        act) + ' recording ' + str(rec)

                    for i in tqdm(range(min(im_mc.shape[1], im_mc2.shape[1]))):
                        try:
                            # Read Depth data from two kinects
                            d1 = misc.imread(
                                '%s/Kinect/Kin01/S%02d/A%02d/R%02d/kin_k01_s%02d_a%02d_r%02d_depth_%05d.pgm' % (
                                    drc, sub, act, rec, sub, act, rec, i))
                            d2 = misc.imread(
                                '%s/Kinect/Kin02/S%02d/A%02d/R%02d/kin_k02_s%02d_a%02d_r%02d_depth_%05d.pgm' % (
                                    drc, sub, act, rec, sub, act, rec, i))

                            # read in the color data
                            il1 = misc.imread(
                                '%s/Kin01/S%02d/A%02d/R%02d/kin_k01_s%02d_a%02d_r%02d_depth_%05d.png' % (
                                    dir_labels, sub, act, rec, sub, act, rec, i))
                            il2 = misc.imread(
                                '%s/Kin02/S%02d/A%02d/R%02d/kin_k02_s%02d_a%02d_r%02d_depth_%05d.png' % (
                                    dir_labels, sub, act, rec, sub, act, rec, i))

                            # Apply inverse projection to fetch point clouds in mm and corresponding indices
                            pts1, indices1, mask1 = utils.map_range_2_pc(d1, fx1, fy1, cx1, cy1, h1, center, stand, thresh=5)
                            pts2, indices2, mask2 = utils.map_range_2_pc(d2, fx2, fy2, cx2, cy2, h2, center, stand, thresh=5)

                            idx_mask1, idx_mask2 = (mask1 == 0), (mask2 == 0)

                            il1[idx_mask1], il2[idx_mask2] = 0, 0

                            # get pixel level classes for all the body parts (+ Background)
                            lbl1 = utils.get_pixel_level_classes(il1, mhad_color_map)
                            lbl2 = utils.get_pixel_level_classes(il2, mhad_color_map)

                            label_idx1 = np.zeros(indices1.shape[0])
                            label_idx2 = np.zeros(indices2.shape[0])

                            for index in range(indices1.shape[0]):
                                label_idx1[index] = lbl1[indices1[index, 1], indices1[index, 0]]

                            for index in range(indices2.shape[0]):
                                label_idx2[index] = lbl2[indices2[index, 1], indices2[index, 0]]

                            full_pc = np.hstack([pts1, pts2])
                            full_pc = full_pc[[1, 2, 0], :]

                            # plotxyz(full_pc.T, color='b', hold=False)

                            skel = skel_jnt.T
                            jnt = np.reshape(skel[int(im_mc[2, i]) + 1, :], (35, 3)).T

                            # plotxyz(jnt.T, color='r', hold=False)

                            label_idx = np.hstack([label_idx1, label_idx2])

                            full_pcl = np.vstack([full_pc, np.reshape(label_idx, (1, label_idx.shape[0]))])

                            # store joint-wise features
                            feature_image = np.array([])

                            for joint in range(jnt.shape[1]):
                                ind_seg = full_pcl[3, :] == joint
                                pcl_joint = full_pcl[0:3, ind_seg]
                                pcl_joint[np.isnan(pcl_joint)] = 0

                                # extract moments
                                med_joint = np.median(pcl_joint, 1) if pcl_joint.size else np.zeros(3)
                                med_joint[np.isnan(med_joint)] = 0

                                std_joint = np.std(pcl_joint, 1) if pcl_joint.size else np.zeros(3)
                                std_joint[np.isnan(std_joint)] = 0

                                min_joint = np.min(pcl_joint, 1) if pcl_joint.size else np.zeros(3)
                                min_joint[np.isnan(min_joint)] = 0

                                max_joint = np.max(pcl_joint, 1) if pcl_joint.size else np.zeros(3)
                                max_joint[np.isnan(max_joint)] = 0

                                cov_joint = np.cov(pcl_joint) if pcl_joint.size else np.zeros((3, 3))
                                cov_joint[np.isnan(cov_joint)] = 0

                                eig_joint = np.linalg.eigvals(cov_joint)
                                eig_joint[np.isnan(eig_joint)] = 0

                                feature_joint = np.concatenate(
                                    [np.ravel(med_joint), np.ravel(std_joint), np.ravel(min_joint),
                                     np.ravel(max_joint), np.ravel(eig_joint), np.ravel(cov_joint),
                                     ], axis=0)

                                feature_image = np.concatenate([np.ravel(feature_image), np.ravel(feature_joint)]) \
                                    if feature_image.size else feature_joint

                            feature = np.vstack([feature, feature_image]) if feature.size else feature_image
                            # print feature.shape
                            gt = np.vstack([gt, np.ravel(jnt)]) if gt.size else np.ravel(jnt)

                        except (IndexError, IOError) as ee:
                            print '\nNumber ' + str(i) + ' image is empty!'

                except IOError:
                    print '\nFile does not exist!'

        feature_dictionary[str(sub)].append(feature)
        pickle.dump(feature_dictionary, pickle_fv_inferred)

        gt_dictionary[str(sub)].append(gt)
        pickle.dump(gt_dictionary, pickle_gt_inferred)

    return feature_dictionary, gt_dictionary


if __name__ == '__main__':



    # Read camera parameters
    config_file = './config_files/camera_params.mat'
    drc = '/media/mcao/Miguel/BerkeleyMHAD/'
    dir_labels = '/media/mcao/Miguel/MHAD/Kinect/TestResults_correct_db/'

    # Limits to parse data directories for ground truth
    # subjects = 12
    # actions = 11
    # recordings = 5

    mhad_color_map = color_map['MHAD']

    # feature_dictionary, gt_dictionary = extract_features_raw(drc, config_file, subjects, actions, recordings)

    # Limits to parse data directories for the labels inferred
    subjects = 12
    actions = 11
    recordings = 5

    feature_dictionary_inferred, gt_dictionary_inferred = extract_features_labelled(drc, config_file, dir_labels, mhad_color_map,
                                                                  subjects, actions, recordings)

    # Saving all the features (train- test)
    # pickle_fv = open("../ckpts/ckpts_mhad_python/features_full_20180208.pickle", "wb")
    # pickle.dump(feature_dictionary, pickle_fv)
    # pickle_fv.close()
    #
    # pickle_gt = open("../ckpts/ckpts_mhad_python/ground_truth_full_20120208.pickle", "wb")
    # pickle.dump(gt_dictionary, pickle_gt)
    # pickle_gt.close()ss
    #
    pickle_fv_inferred.close()
    pickle_gt_inferred.close()
