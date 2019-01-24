import pickle
import glob
import matplotlib.pyplot as plt
from ubc_args import args


if __name__ == '__main__':
  pose_list = sorted(glob.glob('%s*/gt_poses.pkl' % args.root_dir))
  for array_idx, pose_array in enumerate(pose_list):
    pkl_array = pickle.load(open(pose_array, 'rb'))
    for idx in range(1001):
      pose = pkl_array[idx, ]

      # print pose.shape
