import math
import glob
import pickle
import numpy as np
import os
import keras
import cv2

# Change argument source file based on the dataset
from ubc_args import args


class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_ids, batch_size=10, dim=(10, 16, 2),
                 dim_visual=(10, 16, 565), num_actions=3, shuffle=True, flag_data='flow_kron'):

        self.dim = dim
        self.num_actions = num_actions
        self.dim_visual = dim_visual
        self.batch_size = batch_size
        self.list_ids = list_ids
        self.shuffle = shuffle
        self.indexes = []
        self.flag_data = flag_data
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = [], []

        # Find list of IDs
        list_IDs_temp = [self.list_ids[k] for k in indexes]

        # Generate data
        if self.flag_data == 'flow_kron':
            X, y = self.__data_generation_kron(list_IDs_temp, feature_dir=args.root_dir)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation_kron(self, list_ids_temp, feature_dir):
        X = np.empty((self.batch_size, 100, 100, 3), dtype=np.float16)
        y = np.empty((self.batch_size, 54), dtype=np.float16)
        # y = [[None]] * self.batch_size

        for idx, id_name in enumerate(list_ids_temp):
          img = cv2.imread(id_name)
          img = cv2.resize(img, (100, 100))
          X[idx, ] = img.astype(np.float16)
          name_parse = id_name.split('/')
          img_idx = int(name_parse[-1].split('.')[1]) - 1
          vid_idx = name_parse[-5]
          gt_file = '%s%s/gt_poses.pkl' % (feature_dir, vid_idx)
          gt_pose = pickle.load(open(gt_file, 'r'))[img_idx, ]
          gt_pose = np.reshape(gt_pose, (-1, )).astype(np.float16)
          y[idx, ] = gt_pose

        return X, y


if __name__ == '__main__':

  img_list = sorted(glob.glob('%s*/images/depthRender/Cam1/*.png' % args.root_dir))
  qwe = DataGenerator(img_list, flag_data='flow_kron', batch_size=32)
  a, b = qwe.__getitem__(0)
  print a[0].dtype, b[0].shape