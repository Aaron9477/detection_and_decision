#!/usr/bin/env bash

# default convert
#python convert.py yolov3-tiny.cfg yolov3-tiny.weights model_data/yolov3_tiny.h5

# transfer my model from darknet to keras
# python convert.py turtlebot_model/yolov3-tiny.cfg turtlebot_model/yolov3-tiny_final.weights model_data/yolov3_tiny_turtlebot.h5


# test the model trained from darknet, using the video
#python yolo_video.py --model turtlebot_model/yolov3_tiny_turtlebot.h5 --anchors turtlebot_model/anchor.txt --classes turtlebot_model/classes.txt --gpu_num 1 --input /media/zq610/2C9BDE0469DC4DFC/ubuntu/dl_dataset/turtlebot/raw_videos/turtlebot2_1.mp4


# test the model trained from darknet, using the image. There need to input image name. Please attention!! The name of the image must be included by ""
#python yolo_video.py --model turtlebot_model/yolov3_tiny_turtlebot.h5 --anchors turtlebot_model/anchor.txt --classes turtlebot_model/classes.txt --gpu_num 0 --image


# convert backbone weight from darknet to h5(keras). This cfg file has been changed to adapt the backbone network with 15 layers.
#python convert.py /home/zq610/WYZ/deeplearning/network/keras-yolo3/turtlebot_training/yolov3-tiny_15.cfg /home/zq610/WYZ/deeplearning/network/keras-yolo3/turtlebot_training/yolov3-tiny_15.weights  turtlebot_training/darknet15_weights2.h5

# train
# There must use python3, because python2 has some problems
#python3 train.py


# test the model trained from keras-yolo, using the video
#python yolo_video.py --model logs/trained_weights_final.h5 --anchors turtlebot_training/anchor.txt --classes turtlebot_training/classes.txt --gpu_num 1 --input /media/zq610/2C9BDE0469DC4DFC/ubuntu/dl_dataset/turtlebot/raw_videos/two_small_1.mp4

