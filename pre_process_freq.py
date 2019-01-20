import numpy as np
import cv2 as cv
import config
import glob
import os


def median_freq(ba=[]):
    ba.sort()
    len_list = len(ba)
    if len_list % 2 != 0:
        return ba[len_list / 2]
    elif len_list % 2 == 0:
        med = ((ba[len_list / 2] + ba[(len_list / 2) - 1]) / 2)
        return med


def frequency_balanced_weights(median_frequency, ba1=[]):
    freq_bal_weights = np.divide(np.float32(median_frequency), np.float32(ba1))
    return [float(i)/sum(freq_bal_weights) for i in freq_bal_weights]


data_flag_UBC = False
data_flag_forced_UBC = False
data_flag_MHAD = False
data_flag_UBC_interpolated = True
first_iter = True

if data_flag_UBC:
    train_set = 1
    cam_range = 3
    list_config = config.colors['UBC_easy']
    freq = []
    for a in range(1, train_set + 1):
        for b in range(1, cam_range + 1):
            print 'Saving Images'
            for c in range(0, 1000):
                image = cv.imread(
                    '/media/mcao/Miguel/input/UBC_easy/train/%d/images/groundtruth/Cam%d/mayaProject.%06d.png'
                    % (a, b, (c + 1)))

                qwe = np.array(image).sum(axis=2)
                itemidx = np.where(qwe.sum(axis=0) != 0)
                itemidy = np.where(qwe.sum(axis=1) != 0)
                cropped_image = image[min(itemidy[0]):max(itemidy[0]), min(itemidx[0]):max(itemidx[0]), :]
                qqq = cv.resize(cropped_image, (224, 224), interpolation=cv.INTER_NEAREST)

                np_image = np.array(qqq)
                if first_iter:
                    print 'Job is done'
                    for color in list_config:
                        freq.append(np_image[(np_image == color)].shape[0]/3)
                        # print color, np_image[(np_image == color)].shape[0]/3
                    first_iter = False
                else:
                    for i in range(0, 45):
                        freq[i] += np_image[(np_image == list_config[i])].shape[0]/3
                print 'Step %d ' % a, 'Camera %d ' % b, max(freq)

    print freq
    list_a = list(freq)
    list_wt = list(freq)

    median_frequency = median_freq(list_a)
    weights = frequency_balanced_weights(median_frequency, list_wt)
    print freq, '\n', weights, median_frequency
    with open('sanity_frequencies_UBC.txt', 'wb') as f1:
        for item in weights:
            f1.write("%s\n" % item)
    f1.close()


if data_flag_UBC_interpolated:
    train_set = 60
    cam_range = 3
    list_config = config.colors['UBC_interpolated']
    freq = []
    for a in range(1, train_set + 1):
        if not a == 6:
            for b in range(1, cam_range + 1):
                for c in range(0, 1001):
                    image = cv.imread(
                        '/media/mcao/Miguel/UBC_hard/train/%d/images/interpol_groundtruth/Cam%d/mayaProject.%06d.png'
                        % (a, b, (c + 1)))

                    qwe = np.array(image).sum(axis=2)
                    itemidx = np.where(qwe.sum(axis=0) != 0)
                    itemidy = np.where(qwe.sum(axis=1) != 0)
                    cropped_image = image[min(itemidy[0]):max(itemidy[0]), min(itemidx[0]):max(itemidx[0]), :]
                    qqq = cv.resize(cropped_image, (224, 224), interpolation=cv.INTER_NEAREST)

                    np_image = np.array(qqq)
                    if first_iter:
                        print 'Job is done'
                        for color in list_config:
                            freq.append(np_image[(np_image == color)].shape[0]/3)
                            # print color, np_image[(np_image == color)].shape[0]/3
                        first_iter = False
                    else:
                        for i in range(0, 30):
                            freq[i] += np_image[(np_image == list_config[i])].shape[0]/3
                print 'Step %d ' % a, 'Camera %d ' % b, max(freq)

    print freq
    list_a = list(freq)
    list_wt = list(freq)

    median_frequency = median_freq(list_a)
    weights = frequency_balanced_weights(median_frequency, list_wt)
    print freq, '\n', weights, median_frequency
    with open('frequencies_UBC_interpolated.txt', 'wb') as f1:
        for item in weights:
            f1.write("%s\n" % item)
    f1.close()

if data_flag_forced_UBC:
    train_set = 60
    cam_range = 3
    list_config = config.colors['forced_UBC_easy']
    freq = []
    for a in range(1, train_set + 1):
        if not a == 6:
            for b in range(1, cam_range + 1):
                for c in range(0, 1001):
                    image = cv.imread(
                        '/media/mcao/Miguel/UBC_hard/train/%d/images/forced_groundtruth/Cam%d/mayaProject.%06d.png'
                        % (a, b, (c + 1)))

                    qwe = np.array(image).sum(axis=2)
                    itemidx = np.where(qwe.sum(axis=0) != 0)
                    itemidy = np.where(qwe.sum(axis=1) != 0)
                    cropped_image = image[min(itemidy[0]):max(itemidy[0]), min(itemidx[0]):max(itemidx[0]), :]
                    qqq = cv.resize(cropped_image, (224, 224), interpolation=cv.INTER_NEAREST)

                    np_image = np.array(qqq)
                    if first_iter:
                        print 'Job is done'
                        for color in list_config:
                            freq.append(np_image[(np_image == color)].shape[0]/3)
                            # print color, np_image[(np_image == color)].shape[0]/3
                        first_iter = False
                    else:
                        for i in range(0, 19):
                            freq[i] += np_image[(np_image == list_config[i])].shape[0]/3
                print 'Step %d ' % a, 'Camera %d ' % b, max(freq)

    print freq
    list_a = list(freq)
    list_wt = list(freq)

    median_frequency = median_freq(list_a)
    weights = frequency_balanced_weights(median_frequency, list_wt)
    print freq, '\n', weights, median_frequency
    with open('frequencies_forced_UBC_hard_sanity.txt', 'wb') as f1:
        for item in weights:
            f1.write("%s\n" % item)
    f1.close()


if data_flag_MHAD:
    main_input_dir = '/media/mcao/Miguel/MHAD/Kinect/'
    cameras = 2
    subjects = 1
    actions = 1
    recordings = 1

    list_config = config.colors['MHAD']
    freq = []

    for cam in range(1, cameras + 1):
        for sub in range(1, subjects + 1):
            for act in range(1, actions + 1):
                for rec in range(1, recordings + 1):
                    label_file_list = glob.glob(os.path.join(main_input_dir, 'Kin%02d/S%02d/A%02d/R%02d/labels'
                                                             % (cam, sub, act, rec), '*.png'))
                    print 'Saving Images'
                    for label_image in label_file_list:

                        try:
                            image = cv.imread(label_image)
                            qwe = np.array(image).sum(axis=2)
                            itemidx = np.where(qwe.sum(axis=0) != 0)
                            itemidy = np.where(qwe.sum(axis=1) != 0)
                            cropped_image = image[min(itemidy[0]):max(itemidy[0]), min(itemidx[0]):max(itemidx[0]), :]
                            resized_image = cv.resize(cropped_image, (224, 224), interpolation=cv.INTER_NEAREST)

                            np_image = np.array(resized_image)
                            if first_iter:
                                print 'Job is done'
                                for color in list_config:
                                    freq.append(np_image[(np_image == color)].shape[0] / 3)

                                first_iter = False
                            else:
                                for i in range(0, 36):
                                    freq[i] += np_image[(np_image == list_config[i])].shape[0] / 3
                            print 'Camera %d ' % cam, 'Subject %d ' % sub, max(freq)

                        except:
                            print('Missed the image  %s' % label_image)

    print freq
    list_a = list(freq)
    list_wt = list(freq)

    median_frequency = median_freq(list_a)
    weights = frequency_balanced_weights(median_frequency, list_wt)
    print freq, '\n', weights, median_frequency
    with open('frequencies_MHAD.txt', 'wb') as f1:
        for item in weights:
            f1.write("%s\n" % item)
    f1.close()