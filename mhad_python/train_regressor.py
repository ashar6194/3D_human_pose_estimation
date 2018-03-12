from __future__ import absolute_import, division
import h5py
import hdf5storage
import numpy as np
import os
from sklearn import linear_model
import pickle

pkl_path = '/media/mcao/Miguel/UBC/ckpts/ckpts_Sample_net3/ridge/'
mat_dir = '/media/mcao/Miguel/MHAD/Mat_files/February/UBC/UBC_Sample_net3/'

mat_file_fs = mat_dir + 'feature_net3_original_sample_tr.mat'
mat_file_gt = mat_dir + 'gt_net3_original_sample_tr.mat'
num_models = 54


def feature_set_xcorr():
    index = np.array([])
    for i in range(0, 840, 24):
        new_index = np.concatenate([np.arange(i, i + 3), np.arange(i + 6, i + 12)])
        index = np.concatenate([index, new_index]) if index.size else new_index
    index = np.concatenate([index, np.arange(i + 24, i + 233)])
    return index


def feature_set_without_xcorr():
    return np.arange(1080)


def feature_set_full():
    return np.arange(1049)


# training_features = sio.loadmat(mat_file)
feature_training = h5py.File(mat_file_fs, 'r')
gt_training = h5py.File(mat_file_gt, 'r')

# Dereference the h5py objects
object_feature = feature_training.get(feature_training.keys()[1].encode('ascii'))
object_gt = gt_training.get(gt_training.keys()[1].encode('ascii'))

train_features = np.array([])
ground_truth = np.array([])

for i in range(0, 9):
    print 'training with the subject number ' + str(i)
    subject_wise_feature = np.asarray(feature_training[object_feature[0][i]])
    subject_wise_gt = np.asarray(gt_training[object_gt[0][i]])
    train_features = np.vstack([train_features, subject_wise_feature]) \
        if train_features.size else subject_wise_feature
    ground_truth = np.vstack([ground_truth, subject_wise_gt]) \
        if ground_truth.size else subject_wise_gt

print 'Shape of feature set', train_features.shape
print 'Shape of ground truth', ground_truth.shape


func_dictionary = {
    'a': feature_set_xcorr,
    'b': feature_set_without_xcorr,
    'c': feature_set_full,
}

idx = func_dictionary['b']()
truncated_train_features = train_features[:, idx]
print truncated_train_features.shape

alpha = [0.75, 1.25]

# The prediction pipeline
mat_file_fs = mat_dir + 'feature_net3_original_sample_ts.mat'
feature_testing = h5py.File(mat_file_fs, 'r')

# Dereference the h5py objects
object_feature = feature_testing.get(feature_testing.keys()[1].encode('ascii'))

test_features = np.array([])
func_dictionary = {
    'a': feature_set_xcorr,
    'b': feature_set_without_xcorr,
    'c': feature_set_full,
}

for i in range(3):
    print 'predicting for subject number ' + str(i)
    subject_wise_feature = np.asarray(feature_testing[object_feature[0][i]])
    subject_wise_gt = np.asarray(gt_training[object_gt[0][i]])
    test_features = np.vstack([test_features, subject_wise_feature]) \
        if test_features.size else subject_wise_feature

print 'Shape of test feature set', test_features.shape


idx = func_dictionary['b']()
truncated_test_features = test_features[:, idx]

print truncated_test_features.shape

for alp in alpha:
    for i in range(num_models):
        mdl = linear_model.Ridge(alpha=alp)
        # print ('Currently processing joint number- ' + str(i))
        Y = ground_truth[:, i]
        train_labels = Y.reshape(Y.shape[0], 1)
        mdl.fit(truncated_train_features , train_labels)
        # pkl_file = './ckpts/ckpts_co_regression/model_co_%d.pkl' % i
        if not os.path.exists(pkl_path):
            os.makedirs(pkl_path)
        pkl_file = pkl_path + 'model%d.pkl' % i
        with open(pkl_file, 'wb') as model_file:
            pickle.dump(mdl, model_file)

    predicted_data = np.array([])

    for i in range(num_models):
        pkl_file = pkl_path + '/model%d.pkl' % i
        mdl = pickle.load(open(pkl_file, 'rb'))
        prediction_single = mdl.predict(truncated_test_features)
        predicted_data = np.hstack([predicted_data, prediction_single]) if predicted_data.size else prediction_single

    print 'The final shape of predicted data is ' + str(predicted_data.shape)
    file_name = 'skl_pred_ubc_net3_original_sample_ridge%d.mat' % (alp * 100)
    hdf5storage.write(np.transpose(predicted_data), path ='/tf_predicted_data',
                      filename=mat_dir + file_name,
                      matlab_compatible=True)
