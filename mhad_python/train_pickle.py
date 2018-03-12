import pickle
import numpy as np
from sklearn import linear_model
import os

pickle_feature_file = open("../ckpts/ckpts_mhad_python/features_full_20180208.pickle", "rb")
feature = pickle.load(pickle_feature_file)

pickle_gt_file = open("../ckpts/ckpts_mhad_python/ground_truth_full_20120208.pickle", "rb")
gt = pickle.load(pickle_gt_file)

# select idx for the features to be used
index = np.array([])
for i in range(0, 840, 24):
    new_index = np.concatenate([np.arange(i, i+3), np.arange(i+12, i+21)])
    index = np.concatenate([index, new_index]) if index.size else new_index
# print index

features = np.array([])
gts = np.array([])

for i in range(2, 13):
    feature_set = np.sum(np.array(feature[str(i)]), axis=0)
    features = np.concatenate([features, feature_set]) if features.size else feature_set
    gt_set = np.sum(np.array(gt[str(i)]), axis=0)
    gts = np.concatenate([gts, gt_set]) if gts.size else gt_set
    # print np.count_nonzero(gt_set.imag != 0), np.count_nonzero(feature_set.imag != 0)

truncated_features = features #[:, index]
print gts.shape, truncated_features.shape

for i in range(105):
    mdl = linear_model.LinearRegression()
    # print ('Currently processing joint number- ' + str(i))
    Y = gts[:, i]
    train_labels = Y.reshape(Y.shape[0], 1)
    mdl.fit(truncated_features, train_labels)
    pkl_path = '../ckpts/ckpts_regression/full_20160208/'
    if not os.path.exists(pkl_path):
        os.mkdir(pkl_path)
    pkl_file = pkl_path + 'model%d.pkl' % i

    with open(pkl_file, 'wb') as model_file:
        pickle.dump(mdl, model_file)

