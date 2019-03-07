import gflags



FLAGS = gflags.FLAGS

# Input
gflags.DEFINE_integer('img_width', 480, 'Target Image Width')
gflags.DEFINE_integer('img_height', 360, 'Target Image Height')

gflags.DEFINE_integer('crop_img_width', 320, 'Cropped image widht')
gflags.DEFINE_integer('crop_img_height', 320, 'Cropped image height')

gflags.DEFINE_string('img_mode', "rgb", 'Load mode for images, either '
                     'rgb or grayscale')

# Training
gflags.DEFINE_integer('batch_size', 64, 'Batch size in training and evaluation')
gflags.DEFINE_integer('epochs_first', 15, 'Number of epochs for training')
gflags.DEFINE_integer('epochs_second', 30, 'Number of epochs for training')
gflags.DEFINE_integer('log_rate', 10, 'Logging rate for full model (epochs)')
gflags.DEFINE_integer('initial_epoch',0 , 'Initial epoch to start training')

# Files
gflags.DEFINE_string('experiment_rootdir', "/home/zq610/WYZ/deeplearning/network/keras-yolo3/turtlebot_model", 'Folder '
                     ' containing all the logs, model weights and results')
gflags.DEFINE_string('train_dir', "/media/zq610/2C9BDE0469DC4DFC/ubuntu/dl_dataset/Dornet/training", 'Folder containing'
                     ' training experiments')
gflags.DEFINE_string('val_dir', "/media/zq610/2C9BDE0469DC4DFC/ubuntu/dl_dataset/Dornet/validation", 'Folder containing'
                     ' validation experiments')
gflags.DEFINE_string('test_dir', "/media/zq610/2C9BDE0469DC4DFC/ubuntu/dl_dataset/Dornet/testing", 'Folder containing'
                     ' testing experiments')
gflags.DEFINE_string('log_dir', "/home/zq610/WYZ/deeplearning/network/keras-yolo3/logs/detection_decision", 'Folder containing'
                     ' training logs')

# Model
gflags.DEFINE_bool('restore_model', False, 'Whether to restore a trained'
                   ' model for training')
gflags.DEFINE_string('weights_fname', "yolov3_tiny_turtlebot.h5", '(Relative) '
                                          'filename of model weights')
gflags.DEFINE_string('json_model_fname', "model_struct.json",
                          'Model struct json serialization, filename')

