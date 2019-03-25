#! /usr/bin/env python
#!coding=utf-8
"""
Retrain the YOLO model for your own dataset.
"""

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss, tiny_yolo_body_changed
from yolo3.utils import get_random_data
import sys
import time
import argparse


def _main(argv):
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        '--log_dir', type=str, help='path to model weight file',
        default='train_result/yolo_voc/yolo_changed_new_anchor_3.20/'
    )

    parser.add_argument(
        '--anchors_path', type=str, help='path to anchor definitions',
        default='train_setting/voc_train_data/kmeans_changed_anchors.txt'
    )
    args = parser.parse_args()

    # change the setting following!!!#############
    annotation_path = 'train_setting/voc_train_data/2007_train.txt'
    log_dir = args.log_dir
    classes_path = 'train_setting/origin_tiny_yolo/voc_classes.txt'
    anchors_path = args.anchors_path
    # only one of the following will be used
    tiny_yolo_weights_path = 'train_setting/voc_train_data/darknet15_weights.h5'
    yolo_weights_path = 'model_data/yolo_weights.h5'
    first_stage_bs = 16
    second_stage_bs = 16
    ##############################################

    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (320, 320) # multiple of 32, hw

    is_tiny_version = len(anchors)==6 # default setting
    if is_tiny_version:
        model = create_tiny_model_changed(input_shape, anchors, num_classes,
            freeze_body=3, weights_path=tiny_yolo_weights_path)
        # model = create_tiny_model(input_shape, anchors, num_classes,
        #     freeze_body=3, weights_path=tiny_yolo_weights_path)
    else:
        model = create_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path=yolo_weights_path) # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_dir)
    # 该回调函数将在period个epoch后保存模型到path中
    checkpoint_stage1 = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=5)
    checkpoint_stage2_best = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
    checkpoint_stage2_interval = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=False, period=5)
    # 当评价指标不在提升时，减少学习率
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=15, verbose=1)
    # 当监测值不再改善时，该回调函数将中止训练
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=40, verbose=1)

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        # 把y_true当成输入，作为模型的多输入，把loss封装为层（即Lambda层），作为模型的输出
        # 在模型中，最终输出的y_pred就是loss
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = first_stage_bs
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=50,
                initial_epoch=0,
                callbacks=[logging, checkpoint_stage1])
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = second_stage_bs # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=300,
            initial_epoch=50,
            callbacks=[logging, checkpoint_stage2_best, checkpoint_stage2_interval, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_final.h5')

    # Further training if needed.


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    # {0:32, 1:16, 2:8}[l]这里是使用dict,根据l取参数
    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss', \
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})([*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def create_tiny_model_changed(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    # shape=(?, 10, 10, 3, 7) dtype=float32>, <tf.Tensor 'input_3:0' shape=(?, 20, 20, 3, 7) dtype=float32>
    # 因为输出有两层，每一层是一个尺度上的检测，num_anchors//2代表这层的anchor的数量
    # 这里是一个字典，0代表32，1代表16
    # num_classes+5，中的5是两个坐标和objectness
    # 这里只是定义了y_true的结构形式，并没有赋值
    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body_changed(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        # by_name=True意味着通过layer的名字进行读取
        # by_name=False意味着通过网络结构进行读取
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2, 3]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2, 31)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    # model_body.load_weights('/home/zq610/WYZ/deeplearning/network/keras-yolo3/train_result/yolo_voc/yolo_changed_old_anchor_3.19/ep045-loss16.626-val_loss16.815.h5', by_name=True, skip_mismatch=True)

    # Lambda用以对上一层的输出施以任何TensorFlow表达式,此处是yolo_loss这个函数,[*model_body.output, *y_true]是传入的元素
    # 调用函数使用＊，是将每个元素作为位置参数传入，所以现在相当于传入一个list [*model_body.output, *y_true]，元素数为len(model_body.output)+len(y_true)
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    # shape=(?, 10, 10, 3, 7) dtype=float32>, <tf.Tensor 'input_3:0' shape=(?, 20, 20, 3, 7) dtype=float32>
    # 因为输出有两层，每一层是一个尺度上的检测，num_anchors//2代表这层的anchor的数量
    # 这里是一个字典，0代表32，1代表16
    # num_classes+5，中的5是两个坐标和objectness
    # 这里只是定义了y_true的结构形式，并没有赋值
    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        # by_name=True意味着通过layer的名字进行读取
        # by_name=False意味着通过网络结构进行读取
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2, 3]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2, 31)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    # Lambda用以对上一层的输出施以任何TensorFlow表达式,此处是yolo_loss这个函数,[*model_body.output, *y_true]是传入的元素
    # 调用函数使用＊，是将每个元素作为位置参数传入，所以现在相当于传入一个list [*model_body.output, *y_true]，元素数为len(model_body.output)+len(y_true)
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        # 循环取batch_size数量的数据，所有数据取过一遍后重新打乱
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

if __name__ == '__main__':
    _main(sys.argv)
