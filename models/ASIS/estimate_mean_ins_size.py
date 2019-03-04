# To estimate the mean instance size of each class in training set
import os
import sys
import numpy as np
from scipy import stats
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import provider

parser = argparse.ArgumentParser()
parser.add_argument('--test_area', type=int, default=6, help='The areas except this one will be used to estimate the mean instance size')
FLAGS = parser.parse_args()

def estimate(area):
    LOG_DIR = 'log{}'.format(area)
    num_classes = 13
    file_path = "data/train_hdf5_file_list_woArea{}.txt".format(area)

    train_file_list = provider.getDataFiles(file_path) 

    mean_ins_size = np.zeros(num_classes)
    ptsnum_in_gt = [[] for itmp in range(num_classes)]

    train_data = []
    train_group = []
    train_sem = []
    for h5_filename in train_file_list:
        cur_data, cur_group, _, cur_sem = provider.loadDataFile_with_groupseglabel_stanfordindoor(h5_filename)
        cur_data = np.reshape(cur_data, [-1, cur_data.shape[-1]])
        cur_group = np.reshape(cur_group, [-1])
        cur_sem = np.reshape(cur_sem, [-1])

        un = np.unique(cur_group)
        for ig, g in enumerate(un):
            tmp = (cur_group == g)
            sem_seg_g = int(stats.mode(cur_sem[tmp])[0])
            ptsnum_in_gt[sem_seg_g].append(np.sum(tmp))

    for idx in range(num_classes):
        mean_ins_size[idx] = np.mean(ptsnum_in_gt[idx]).astype(np.int)

    print(mean_ins_size)
    np.savetxt(os.path.join(LOG_DIR, 'mean_ins_size.txt'),mean_ins_size)


if __name__ == "__main__":
    estimate(FLAGS.test_area)
