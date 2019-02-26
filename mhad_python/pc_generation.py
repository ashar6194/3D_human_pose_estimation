from __future__ import absolute_import, division
from tqdm import *
# from view_pc import plotxyz
import pickle
import numpy as np
from scipy import misc
# import h5py
import os
from config import colors as color_map
from utils import get_camera_params, map_range_2_pc, get_skeleton_info, plot_basic_object


OUT_PC_DIR = '/data/MHAD/train_pc/'
OUT_GT_DIR = '/data/MHAD/train_pcgt/'

if not os.path.exists(OUT_PC_DIR):
    os.makedirs(OUT_PC_DIR)
if not os.path.exists(OUT_GT_DIR ):
    os.makedirs(OUT_GT_DIR)


def extract_features_labelled(drc, config_file, dir_labels, mhad_color_map, subjects, actions, recordings):

    # Fetch Camera parameters
    fx1, fy1, cx1, cy1, fx2, fy2, cx2, cy2, h1, h2 = get_camera_params(config_file)

    for sub in range(1, subjects + 1):
        for act in range(1, actions + 1):
            for rec in range(1, recordings + 1):
                try:
                    skeleton_file = drc + 'Skeletons/skl_s%02d_a%02d_r%02d.mat' % (sub, act, rec)
                    im_mc, im_mc2, skel_jnt, center, stand = get_skeleton_info(skeleton_file)

                    print '\nExtraction of PCL in subject ' + str(sub) + ' action ' + str(
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

                            # Apply inverse projection to fetch point clouds in mm and corresponding indices
                            pts1, indices1, mask1 = map_range_2_pc(d1, fx1, fy1, cx1, cy1, h1, center, stand, thresh=5)
                            pts2, indices2, mask2 = map_range_2_pc(d2, fx2, fy2, cx2, cy2, h2, center, stand, thresh=5)

                            full_pc = np.hstack([pts1, pts2])
                            full_pc = full_pc[[1, 2, 0], :]

                            skel = skel_jnt.T
                            jnt = np.reshape(skel[int(im_mc[2, i]) + 1, :], (35, 3)).T
                            # plot_basic_object(full_pc, jnt)

                            out_pc_name = os.path.join(OUT_PC_DIR, 'sub_%02d_act_%02d_rec_%02d_cap_%05d.pkl' % (sub, act, rec, i))
                            out_gt_name = os.path.join(OUT_GT_DIR, 'sub_%02d_act_%02d_rec_%02d_cap_%05d.pkl' % (sub, act, rec, i))

                            pickle.dump(full_pc, open(out_pc_name, 'wb'))
                            pickle.dump(out_gt_name, open(out_gt_name, 'wb'))

                        except (IndexError, IOError) as ee:
                            print '\nNumber ' + str(i) + ' image is empty!'

                except IOError:
                    print '\nFile does not exist!'

    return 0


if __name__ == '__main__':

    # Read camera parameters
    config_file = './config_files/camera_params.mat'
    drc = '/media/mcao/Miguel/BerkeleyMHAD/'
    dir_labels = '/media/mcao/Miguel/MHAD/Kinect/TestResults_correct_db/'

    mhad_color_map = color_map['MHAD']

    # feature_dictionary, gt_dictionary = extract_features_raw(drc, config_file, subjects, actions, recordings)

    # Limits to parse data directories for the labels inferred
    subjects = 12
    actions = 11
    recordings = 5

    feature_dictionary_inferred, gt_dictionary_inferred = extract_features_labelled(drc, config_file, dir_labels, mhad_color_map,
                                                                  subjects, actions, recordings)
