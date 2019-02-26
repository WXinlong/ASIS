import os
import glob

h5_dir = "indoor3d_ins_seg_hdf5/"

for i in range(6):

    filter_id = i + 1

    # For training
    train_fold_pattern = "Area_[!{}]*.h5".format(filter_id)
    dst_list_path = "train_hdf5_file_list_woArea{}.txt".format(filter_id)

    h5_files = glob.glob(os.path.join(h5_dir + train_fold_pattern))

    f = open(dst_list_path, 'w')
    for h5_file in h5_files:
        f.write('data/' + h5_file)
        f.write('\n')
    f.close()

    # For test
    test_fold_pattern = "Area_{}*.h5".format(filter_id)
    dst_list_path = "test_hdf5_file_list_Area{}.txt".format(filter_id)

    h5_files = glob.glob(os.path.join(h5_dir + test_fold_pattern))

    f = open(dst_list_path, 'w')
    for h5_file in h5_files:
        f.write('data/' + h5_file)
        f.write('\n')
    f.close()

