i=$1
python train.py --gpu 0 --batch_size 24 --max_epoch 100  --log_dir log${i} --learning_rate 0.001 --decay_step 300000  --restore_model None --input_list data/train_hdf5_file_list_woArea${i}.txt
python estimate_mean_ins_size.py --test_area ${i}
python test.py --gpu 0 --bandwidth 0.6 --log_dir log${i}_test --model_path log${i}/epoch_99.ckpt --input_list  meta/area${i}_data_label.txt --verbose  
