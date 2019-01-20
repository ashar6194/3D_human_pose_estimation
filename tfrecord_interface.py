from config import get_csv_filename
import json
import os
import config
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='UBC_medium')
args = parser.parse_args()


def generate_csv(output_directory, filename):
    csv = open(filename, 'w')

    if config.working_dataset == 'MHAD':
        columnTitleRow = "TFR_ID,Sub_ID,Action_ID,Rec_ID,Filename_k1,Filename_k2\n"
        csv.write(columnTitleRow)
        cameras = 2
        subjects = 12
        actions = 11
        recordings = 5
        tfr_id = 1

        for sub in range(1, subjects + 1):
            for act in range(1, actions + 1):
                for rec in range(1, recordings + 1):
                    op_file = []
                    for cam in range(1, cameras + 1):

                        strname = 'Kin%02d_s%02d_a%02d_r%02d' % (cam, sub, act, rec)
                        op_file.append(os.path.join(output_directory, '%s.tfrecords' % strname))
                    if os.path.isfile(op_file[0]) and os.path.isfile(op_file[1]):
                        row = str(tfr_id) + "," + str(sub) + "," + str(act) + "," \
                              + str(rec) + "," + op_file[0] + "," + op_file[1] + "\n"
                        csv.write(row)
                        tfr_id += 1

    if config.working_dataset == 'forced_UBC_easy' or config.working_dataset == 'UBC_easy' \
        or config.working_dataset == 'UBC_interpolated':
        columnTitleRow = "TFR_ID,Sub_ID,Cam_ID,Filename\n"
        csv.write(columnTitleRow)
        train_subjects = 60
        valid_subjects = 19
        cameras = 3
        tfr_id = 1

        for sub in range(1, train_subjects + 1):
            for cam in range(1, cameras + 1):
                strname = 'train%d_Cam%d' % (sub, cam)
                op_file = os.path.join(output_directory, '%s.tfrecords' % strname)
                if os.path.isfile(op_file):
                    row = str(tfr_id) + "," + str(sub) + "," + str(cam) + ","  + op_file + "\n"
                    csv.write(row)
                    tfr_id += 1

        for sub in range(1, valid_subjects + 1):
            for cam in range(1, cameras + 1):
                strname = 'valid%d_Cam%d' % (sub, cam)
                op_file = os.path.join(output_directory, '%s.tfrecords' % strname)
                if os.path.isfile(op_file):
                    row = str(tfr_id) + "," + str(sub) + "," + str(cam) + ","  + op_file + "\n"
                    csv.write(row)
                    tfr_id += 1

#
# def generate_json(output_directory, json_dir):
#
#     if config.working_dataset == 'UBC_easy':
#         train_set = 4
#         cam_range = 3
#         path_dictionary = {'train': [], 'val': [], 'test': []}
#         error_dictionary = {'train': [], 'val': [], 'test': []}
#
#         for a in range(1, train_set + 1):
#             for b in range(1, cam_range + 1):
#                 strname = 'train%d_Cam%d' % (a, b)
#                 op_file = os.path.join(output_directory, '%s.tfrecords' % strname)
#                 try:
#                     path_dictionary['train'].append(op_file)
#                 except:
#                     error_dictionary['train'].append(op_file)
#                 print ('Converting Train Example Number: %d, Camera %d' % (a, b))
#
#         if not os.path.exists(json_dir):
#             os.mkdir(json_dir)
#
#         with open('%s/train_sample10k_ubc_original.json' % json_dir, 'w') as outfile:
#             json.dump(path_dictionary, outfile)
#         with open('%s/except_sample10k_ubc_original.json' % json_dir, 'w') as outfile2:
#             json.dump(error_dictionary, outfile2)
#
#     if config.working_dataset == 'MHAD':
#         cameras = 2
#         subjects = 12
#         actions = 11
#         recordings = 5
#
#         path_dictionary = {'train': dict((k, []) for k in range(1, subjects + 1))}
#         error_dictionary = {'train': dict((k, []) for k in range(1, subjects + 1))}
#
#         for cam in range(1, cameras + 1):
#             for sub in range(1, subjects + 1):
#                 for act in range(1, actions + 1):
#                     for rec in range(1, recordings + 1):
#                         strname = 'Kin%02d_s%02d_a%02d_r%02d' % (cam, sub, act, rec)
#                         op_file = os.path.join(output_directory, '%s.tfrecords' % strname)
#                         try:
#                             path_dictionary['train'][sub].append(op_file)
#                         except:
#                             error_dictionary['train'][sub].append(op_file)
#
#         with open('%s/training_mhad_fitted_planev2.json' % json_dir, 'w') as outfile:
#             json.dump(path_dictionary, outfile)
#         with open('%s/error_mhad_fitted_planev2.json' % json_dir, 'w') as outfile2:
#             json.dump(error_dictionary, outfile2)


if __name__ == '__main__':

    main_dir_mhad = '/media/mcao/Miguel/MHAD_fitted_plane/Kinect/'
    main_dir_ubc = '/media/mcao/Miguel/UBC_hard/'

    json_dir = './json'
    csv_dir = './csv'

    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    if args.dataset_name == 'UBC_easy':

        main_dir_ubc = '/media/mcao/Miguel/input/UBC_easy/'
        tfrecord_path = 'TFrecords_original'
        output_trn = 'Train_crop'
        filename = get_csv_filename(args.dataset_name)
        output_directory = os.path.join(main_dir_ubc, tfrecord_path, output_trn)
        generate_csv(output_directory, filename)

    if args.dataset_name == 'UBC_medium':
        print 'Hey!'
        main_dir_ubc = '/media/mcao/Miguel/UBC_medium/'
        tfrecord_path = 'TFrecords'
        output_trn = 'Train'
        filename = get_csv_filename(args.dataset_name)
        output_directory = os.path.join(main_dir_ubc, tfrecord_path, output_trn)
        generate_csv(output_directory, filename)

    if args.dataset_name == 'UBC_hard':
        print 'Hey Hard!'
        tfrecord_path = 'TFrecords'
        output_trn = 'Train'
        filename = get_csv_filename(args.dataset_name)
        output_directory = os.path.join(main_dir_ubc, tfrecord_path, output_trn)
        generate_csv(output_directory, filename)

    elif args.dataset_name == 'forced_UBC_easy':
        tfrecord_path = 'Forced_TFrecords'
        output_trn = 'Train'
        filename = get_csv_filename(args.dataset_name)
        output_directory = os.path.join(main_dir_ubc, tfrecord_path, output_trn)
        generate_csv(output_directory, filename)

    elif args.dataset_name == 'MHAD':
        tfrecord_path = 'TFrecords_fittedplane'
        output_trn = 'Train'
        filename = get_csv_filename(args.dataset_name)
        output_directory = os.path.join(main_dir_mhad, tfrecord_path, output_trn)
        generate_csv(output_directory, filename)

    elif args.dataset_name == 'UBC_interpolated':
        tfrecord_path = 'MHAD_learned_TFrecords'
        output_trn = 'Train'
        filename = get_csv_filename(args.dataset_name)
        output_directory = os.path.join(main_dir_ubc, tfrecord_path, output_trn)
        generate_csv(output_directory, filename)

