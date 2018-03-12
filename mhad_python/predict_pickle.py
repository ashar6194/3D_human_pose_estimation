import pickle
import numpy as np
import matplotlib.pyplot as plt


def box_plot_error(prediction, ground_truth):
    error_full_set = np.sqrt(np.sum(np.reshape(((prediction - ground_truth) ** 2), (3, 35 * (ground_truth.shape[0]))),
                                    axis=0))
    plt.figure()
    plt.subplot(211)
    plt.boxplot(error_full_set)
    plt.grid()
    plt.interactive(False)
    plt.ylabel('Error in mm')
    plt.title('Total Test Error')

    error_joint_wise = np.sqrt(np.sum(np.reshape(((prediction - ground_truth) ** 2), (35, 3,  (ground_truth.shape[0]))),
                                      axis=1))
    error_joint_wisel = error_joint_wise.tolist()
    plt.subplot(212)
    plt.boxplot(error_joint_wisel)
    plt.grid()
    plt.interactive(False)
    plt.ylabel('Error in mm')
    plt.title('Per-Joint Error')
    plt.show()


pickle_feature_file = open("../ckpts/ckpts_mhad_python/features_inferred_s2.pickle", "rb")
feature = pickle.load(pickle_feature_file)

pickle_gt_file = open("../ckpts/ckpts_mhad_python/ground_truth_inferred_s2.pickle", "rb")
gt = pickle.load(pickle_gt_file)


# select idx for the features to be used
index = np.array([])

for i in range(0, 840, 24):
    new_index = np.concatenate([np.arange(i, i+3), np.arange(i+12, i+21)])
    index = np.concatenate([index, new_index]) if index.size else new_index

# Declare the test ground truth and testing features
features = np.array([])
gts = np.array([])

for i in range(1, 2):
    feature_set = np.sum(np.array(feature[str(i)]), axis=0)
    features = np.concatenate([features, feature_set]) if features.size else feature_set
    gt_set = np.sum(np.array(gt[str(i)]), axis=0)
    gts = np.concatenate([gts, gt_set]) if gts.size else gt_set
    # print np.count_nonzero(gt_set.imag != 0), np.count_nonzero(feature_set.imag != 0)

truncated_features = features  # [:, index]
print gts.shape, truncated_features.shape

predicted_data = np.array([])
for i in range(105):
    pkl_file = '../ckpts/ckpts_regression/full_feature_regression/model%d.pkl' % i
    mdl = pickle.load(open(pkl_file, 'rb'))
    prediction_single = mdl.predict(truncated_features)
    predicted_data = np.hstack([predicted_data, prediction_single]) if predicted_data.size else prediction_single

# Calculations for box plots
box_plot_error(predicted_data, gts)

pickle_pred = open("../ckpts/ckpts_mhad_python/predictions_s2.pickle", "wb")
pickle.dump(predicted_data, pickle_pred)
pickle_pred.close()
