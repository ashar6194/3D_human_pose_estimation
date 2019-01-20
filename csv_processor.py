import random
import math
import numpy as np


def get_subject_wise_split(train_file_name, csv_dir, sub_id=1):
    train_file = open(train_file_name, 'r')
    train_idx_file = open('%s/%s_train_sub%d_idx.csv' %
                          (csv_dir, (train_file_name.strip().split('.')[-2].split('/')[-1]), sub_id), 'w')
    test_idx_file = open('%s/%s_test_sub%d_idx.csv' %
                         (csv_dir, (train_file_name.strip().split('.')[-2].split('/')[-1]), sub_id), 'w')
    first_row_flag = True
    train_idx, test_idx, filename_list_1, filename_list_2, train_file_list, test_file_list = \
        [], [], [], [], [], []

    for row in train_file:
        if not first_row_flag:
            if not int(row.strip().split(',')[1]) == sub_id:
                train_idx.append(int(row.strip().split(',')[0]))
                train_file_list.extend((row.strip().split(',')[-2], row.strip().split(',')[-1]))
                train_idx_file.write((row.strip().split(',')[0]) + "," + row.strip().split(',')[-2]
                                     + "," + row.strip().split(',')[-1] + '\n')

            else:
                test_idx.append(int(row.strip().split(',')[0]))
                test_file_list.extend((row.strip().split(',')[-2], row.strip().split(',')[-1]))
                test_idx_file.write((row.strip().split(',')[0]) + "," + row.strip().split(',')[-2]
                                     + "," + row.strip().split(',')[-1] + '\n')
        first_row_flag = False

    return train_file_list, test_file_list, train_idx, test_idx


def get_random_percent_split(train_file_name, csv_dir, ratio=0.75):
    random.seed(1)
    train_file = open(train_file_name, 'r')
    train_idx_file = open('%s/%s_train_idx.csv' % (csv_dir, (train_file_name.strip().split('.')[-2].split('/')[-1])), 'w')
    test_idx_file = open('%s/%s_test_idx.csv' % (csv_dir, (train_file_name.strip().split('.')[-2].split('/')[-1])), 'w')
    first_row_flag = True
    tfr_id_list , filename_list_1, filename_list_2, train_file_list, test_file_list = \
        [], [], [], [], []

    for row in train_file:
        if not first_row_flag:
            tfr_id_list.append(int(row.strip().split(',')[0]))
            filename_list_1.append(row.strip().split(',')[-2])
            filename_list_2.append(row.strip().split(',')[-1])
        first_row_flag = False
    random.shuffle(tfr_id_list)
    train_idx = tfr_id_list[0: int(math.floor(ratio * len(tfr_id_list)))]
    test_idx = tfr_id_list[int(math.floor(ratio * len(tfr_id_list))):len(tfr_id_list)]

    for i in train_idx:
        train_file_list.extend((filename_list_1[i - 1], filename_list_2[i - 1]))
        train_idx_file.write(str(i) + "," + filename_list_1[i - 1] + "," + filename_list_2[i - 1] + '\n')

    for j in test_idx:
        test_file_list.extend((filename_list_1[j - 1], filename_list_2[j - 1]))
        test_idx_file.write(str(j) + "," + filename_list_1[j - 1] + "," + filename_list_2[j - 1] + '\n')

    print len(train_idx), len(test_idx), len(tfr_id_list)
    return train_file_list, test_file_list, tfr_id_list


def get_train_list(train_file_name):
    train_file = open(train_file_name, 'r')
    first_row_flag = True
    filename_list = []
    for row in train_file:
        if not first_row_flag:
            filename_list.append(row.strip().split(',')[-1])
        first_row_flag = False
    return filename_list


def get_idx_list(train_file_name):
    train_file = open(train_file_name, 'r')
    idx_list = []
    for row in train_file:
        idx_list.append(int(row.strip().split(',')[0]))
    return np.array(idx_list)
#
#
# csv_dir = './csv'
# train_file_name = '%s/checkmhad.csv' % csv_dir
# #
# # train_file_list, test_file_list, tfr_id_list = get_random_percent_split(train_file_name, csv_dir)
# #
# # # filename_list = get_train_list(train_file_name)
# # print train_file_list
# # print len(train_file_list)
# #
#
# train_file_list, test_file_list, train_idx, test_idx = get_subject_wise_split(train_file_name, csv_dir, 1)
# print len(train_file_list)