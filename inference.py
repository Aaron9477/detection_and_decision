import sys
import argparse
# from yolo_mobilenet import YOLO, detect_video
from yolo import YOLO, detect_video
from PIL import Image
import os

# VOC dataset
# classes = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor',]
# turtlebot dataset
classes = ['t', 'car']

def detect_image_output(yolo, img_dir, file_write):
    image = Image.open(img_dir)
    out_boxes, out_scores, out_classes = yolo.detect_image_output(image)
    output = ''
    for box_index in range(len(out_classes)):
        output += (classes[out_classes[box_index]] + ' ')
        output += (str(out_scores[box_index]) + ' ')
        # the out_boxes seq is: top, left, bottom, right
        # what we need is: left, top, right, bottom
        tmp_box = [out_boxes[box_index][1], out_boxes[box_index][0], out_boxes[box_index][3], out_boxes[box_index][2]]
        output += ' '.join([str(coord) for coord in tmp_box])
        output += '\n'
    file_write.writelines(output.strip())

def detect_img(yolo, pic_dir, target_dir, pic_name):
    image = Image.open(os.path.join(pic_dir, pic_name))
    r_image = yolo.detect_image(image)
    r_image.save(os.path.join(target_dir, 'voc_detection_' + pic_name))

    # r_image.show()
    yolo.close_session()



FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    # # 单张图片输入进行可视化
    # pic_name = '000002.jpg'
    # pic_dir = '/media/zq610/2C9BDE0469DC4DFC/ubuntu/dl_dataset/VOCdevkit/VOC2007/JPEGImages'
    # target_dir = '/home/zq610/WYZ/graduation_project/dection_result'
    # yolo = YOLO(**vars(FLAGS))
    # detect_img(yolo, pic_dir, target_dir, pic_name)
    # exit()



    # 将所有测试集输入，进行结果输出
    out_folder = '/home/zq610/WYZ/deeplearning/evaluation/Object-Detection-Metrics/detections'

    # VOC datester
    # dataset_dir = '/media/zq610/2C9BDE0469DC4DFC/ubuntu/dl_dataset/VOCdevkit/VOC2007'
    # image_ids = open('%s/ImageSets/Main/%s.txt' % (dataset_dir, 'test')).read().strip().split()
    # turtlebot train dataset
    # dataset_dir = '/media/zq610/2C9BDE0469DC4DFC/ubuntu/dl_dataset/turtlebot/VOC_type/turtlebot'
    # image_ids = open('%s/ImageSets/Main/%s.txt' % (dataset_dir, 'trainval')).read().strip().split()

    # turtlebot train dataset
    dataset_dir = '/media/zq610/2C9BDE0469DC4DFC/ubuntu/dl_dataset/turtlebot/VOC_type/turtlebot_test2'
    image_ids = open('%s/ImageSets/Main/%s.txt' % (dataset_dir, 'train_val')).read().strip().split()

    # 对test文件进行读取
    # 定义对象
    yolo = YOLO(**vars(FLAGS))
    count = 0
    for image_id in image_ids:
        pic_dir = '%s/JPEGImages/%s.jpg' % (dataset_dir, image_id)
        out_file = os.path.join(out_folder, '{image}.txt'.format(image=image_id))
        with open(out_file, 'w') as file_write:
            detect_image_output(yolo, pic_dir, file_write)
        count += 1
        if count % 200 == 0:
            print('{num} labels have been written!'.format(num=count))
    print('all finished! There are {sum} labels'.format(sum=count))
    yolo.close_session()