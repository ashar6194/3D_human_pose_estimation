import scipy.io as sio
import pickle
import numpy as np

pickle_gt_file = open("../ckpts/ckpts_mhad_python/ground_truth_inferred_s3.pickle", "rb")
# gt = pickle.load(pickle_gt_file)
#
data = {}
for __ in range(pickle.load(pickle_gt_file)):
    data.update(pickle.load(pickle_gt_file))
pickle_pred_file = open("../ckpts/ckpts_mhad_python/predictions_s2.pickle", "rb")
predictions = pickle.load(pickle_pred_file)
#
# pickle_fv_inferred = open("../ckpts/ckpts_mhad_python/features_inferred_s3.pickle", "rb")
# feature = pickle.load(pickle_pred_file)
# z
ground_truth = np.array([])

for i in range(1, 2):
    gt_set = np.sum(np.array(gt[str(i)]), axis=0)
    ground_truth = np.concatenate([ground_truth, gt_set]) if ground_truth.size else gt_set

# gt_dictionary['1'].append(predictions)
# pred_dictionary['1'].append(ground_truth)

sio.savemat('zzz_ground_truth_20180208.mat', mdict={'gt': ground_truth})
sio.savemat('zzz_predictions_20180208.mat', mdict={'pred': predictions})


