#!/usr/bin/env bash

file_folder='tmp'
log_dir='train_result/yolo_voc/'${file_folder}'/'
val_dir='/home/zq610/WYZ/deeplearning/evaluation/Object-Detection-Metrics/output/'${file_folder}'/'
anchors_path='train_setting/voc_train_data/kmeans_changed_anchors.txt'

echo ${val_dir}
#python3 train.py --log_dir ${log_dir} --anchors_path ${anchors_path}

