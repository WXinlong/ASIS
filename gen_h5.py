import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import data_prep_util
import indoor3d_util


# Constants
data_dir = os.path.join(BASE_DIR, 'data')
indoor3d_data_dir = os.path.join(data_dir, 'stanford_indoor3d_ins.sem')
NUM_POINT = 4096
data_dtype = 'float32'
label_dtype = 'int32'

# Set paths
filelist = os.path.join(BASE_DIR, 'meta/all_data_label.txt')
data_label_files = [os.path.join(indoor3d_data_dir, line.rstrip()) for line in open(filelist)]
output_dir = os.path.join(data_dir, 'indoor3d_ins_seg_hdf5')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_room_filelist = os.path.join(output_dir, 'room_filelist.txt')
fout_room = open(output_room_filelist, 'w')

sample_cnt = 0
for i in range(0, len(data_label_files)):
    data_label_filename = data_label_files[i]
    fname = os.path.basename(data_label_filename).strip('.npy')
    if not os.path.exists(data_label_filename):
        continue
    data, label, inslabel = indoor3d_util.room2blocks_wrapper_normalized(data_label_filename, NUM_POINT, block_size=1.0, stride=0.5,
                                                 random_sample=False, sample_num=None)
    for _ in range(data.shape[0]):
        fout_room.write(os.path.basename(data_label_filename)[0:-4]+'\n')

    sample_cnt += data.shape[0]
    h5_filename = os.path.join(output_dir, '%s.h5' % fname)
    print('{0}: {1}, {2}, {3}'.format(h5_filename, data.shape, label.shape, inslabel.shape))
    data_prep_util.save_h5ins(h5_filename,
                              data,
                              label,
                              inslabel,
                              data_dtype, label_dtype)

fout_room.close()
print("Total samples: {0}".format(sample_cnt))
