#! /usr/bin/env python
#!coding=utf-8
"""
Retrain the YOLO model for your own dataset.
"""

import os
import sys
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss, tiny_yolo_and_decision
from yolo3.utils import get_random_data
import dornet_utils
from darknet_flags import FLAGS
import tensorflow as tf
from yolo3 import utils
import dronet_log_utils


def _main(argv):
    argv = FLAGS(argv)  # parse flags

    # change the setting following!!!#############
    annotation_path = 'detection_decision/train.txt'
    log_dir = 'logs/detection_decision/'
    classes_path = 'detection_decision/classes.txt'
    anchors_path = 'detection_decision/anchor.txt'
    # because I just use the tiny version, so only one of the following will be used
    # read the weight file trained only by yolo
    tiny_yolo_weights_path = 'turtlebot_model/yolov3_tiny_turtlebot.h5'
    yolo_weights_path = 'model_data/yolo_weights.h5'
    # TODO: the bs may need be changed
    batch_size = FLAGS.batch_size

    # origin trainable layer
    # dronet_trainable_layer = (37, 39, 41, 43, 46, 49, 52, 55, 56)
    # new trainable layer
    dronet_trainable_layer = (37, 39, 41, 43, 46, 49, 52, 55, 56)
    ##############################################

    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    img_width, img_height = FLAGS.img_width, FLAGS.img_height
    crop_img_width, crop_img_height = FLAGS.crop_img_width, FLAGS.crop_img_height

    input_shape = (320, 320) # multiple of 32, hw

    initial_epoch = 0
    if not FLAGS.restore_model:
        # In this case weights will start from random
        weights_path = None
    else:
        weights_path = FLAGS.weights_fname
        initial_epoch = FLAGS.initial_epoch

    is_tiny_version = len(anchors) == 6 # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes, \
            freeze_body=2, yolo_weights_path=tiny_yolo_weights_path, weights_path=weights_path)
    else:
        model = create_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path=yolo_weights_path) # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_dir)
    # 该回调函数将在period个epoch后保存模型到path中
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=False, period=3)
    # 当评价指标不在提升时，减少学习率
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    # 当监测值不再改善时，该回调函数将中止训练
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    dronet_log_utils.configure_output_dir(FLAGS.experiment_rootdir)
    saveModelAndLoss = dronet_log_utils.MyCallback(filepath=FLAGS.experiment_rootdir,
                                                   period=FLAGS.log_rate,
                                                   batch_size=FLAGS.batch_size)

    val_split = 0.1
    # 读取dataset，并打乱顺序
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    print("start training dataset preparing!")
    dronet_train_datagen = dornet_utils.DroneDataGenerator(rotation_range = 0.2,
                                             rescale = 1./255,
                                             width_shift_range = 0.2,
                                             height_shift_range=0.2)

    dronet_train_dataset = dronet_train_datagen.flow_from_directory(directory=FLAGS.train_dir,
                                                        target_size=(img_width, img_height),
                                                        crop_size=(crop_img_height, crop_img_width),
                                                        batch_size=batch_size)

    print("start validation dataset preparing!")
    dronet_val_datagen = dornet_utils.DroneDataGenerator(rescale=1./255)

    dronet_val_dataset = dronet_val_datagen.flow_from_directory(FLAGS.val_dir,
                                                        target_size=(img_width, img_height),
                                                        crop_size=(crop_img_height, crop_img_width),
                                                        batch_size=batch_size)

    # Initialize loss weights
    model.alpha = tf.Variable(1, trainable=False, name='alpha', dtype=tf.float32)
    model.beta = tf.Variable(0, trainable=False, name='beta', dtype=tf.float32)
    # Initialize number of samples for hard-mining
    model.k_mse = tf.Variable(batch_size, trainable=False, name='k_mse', dtype=tf.int32)
    model.k_entropy = tf.Variable(batch_size, trainable=False, name='k_entropy', dtype=tf.int32)

    # json_model_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.json_model_fname)
    # dornet_utils.modelToJson(model, json_model_path)

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    # 这里不是随便起名字，需要是网络的最终输出才能作为字典的元素，下面是随意起名的报错
    # ValueError: Unknown entry in loss dictionary: "steer_loss". Only expected the following keys: ['yolo_loss', 'dense_1', 'activation_2']

    for layer in model.layers:
        if not layer.name.startswith('decision_'):
            layer.trainable = False

    if True:
        model.compile(loss={'yolo_loss': lambda y_true, y_pred: y_pred,
                            'decision_steer_output': dornet_utils.hard_mining_mse(model.k_mse),
                            'decision_coll_output': dornet_utils.hard_mining_entropy(model.k_entropy)},
                    optimizer=Adam(lr=1e-3), loss_weights=[0, model.alpha, model.beta])

        print('Train on {} samples, val on {} samples, with batch size {}.'
              .format(dronet_train_dataset.samples, dronet_val_dataset.samples, batch_size))
        steps_per_epoch = int(np.ceil(dronet_train_dataset.samples / batch_size))
        validation_steps = int(np.ceil(dronet_val_dataset.samples / batch_size))

        # callbacks是回调函数，在合适的时候会被调用，callbacks其实是类的list，这些类均继承callback函数
        model.fit_generator(data_generator_wrapper(dronet_train_dataset, lines[:num_train],
                                                   batch_size, input_shape, anchors, num_classes),
                            epochs=FLAGS.epochs_first, steps_per_epoch=steps_per_epoch,
                            callbacks=[saveModelAndLoss, logging, checkpoint],
                            validation_data=data_generator_wrapper(dronet_val_dataset,
                                            lines[num_train:], batch_size, input_shape, anchors, num_classes),
                            validation_steps=validation_steps,
                            initial_epoch=initial_epoch, )

    if True:
        model.compile(loss={'yolo_loss': lambda y_true, y_pred: y_pred,
                            'decision_steer_output': dornet_utils.hard_mining_mse(model.k_mse),
                            'decision_coll_output': dornet_utils.hard_mining_entropy(model.k_entropy)},
                    optimizer=Adam(lr=1e-4), loss_weights=[0, model.alpha, model.beta])

        model.fit_generator(data_generator_wrapper(dronet_train_dataset, lines[:num_train],
                                                   batch_size, input_shape, anchors, num_classes),
                            epochs=FLAGS.epochs_second, steps_per_epoch=steps_per_epoch,
                            callbacks=[saveModelAndLoss, logging, checkpoint, reduce_lr, early_stopping],
                            validation_data=data_generator_wrapper(dronet_val_dataset,
                                            lines[num_train:], batch_size, input_shape, anchors, num_classes),
                            validation_steps=validation_steps,
                            initial_epoch=FLAGS.epochs_first, )

        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.

    # if True:
    #     for i in range(len(model.layers)):
    #         model.layers[i].trainable = True
    #     model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
    #     print('Unfreeze all of the layers.')
    #
    #     batch_size = second_stage_bs # note that more GPU memory is required after unfreezing the body
    #     print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    #     model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
    #         steps_per_epoch=max(1, num_train//batch_size),
    #         validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
    #         validation_steps=max(1, num_val//batch_size),
    #         epochs=100,
    #         initial_epoch=50,
    #         callbacks=[logging, checkpoint, reduce_lr, early_stopping])
    #     model.save_weights(log_dir + 'trained_weights_final.h5')

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

def create_tiny_model(input_shape, anchors, num_classes, weights_path, dronet_trainable_layer=[],
                      load_pretrained=True, freeze_body=2, yolo_weights_path='model_data/tiny_yolo_weights.h5', ):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    h, w = input_shape
    image_input = Input(shape=(h, w, 3))
    num_anchors = len(anchors)

    # shape=(?, 10, 10, 3, 7) dtype=float32>, <tf.Tensor 'input_3:0' shape=(?, 20, 20, 3, 7) dtype=float32>
    # 因为输出有两层，每一层是一个尺度上的检测，num_anchors//2代表这层的anchor的数量
    # 这里是一个字典，0代表32，1代表16
    # num_classes+5，中的5是两个坐标和objectness
    # 这里只是定义了y_true的结构形式，并没有赋值
    # 这里的y_true是作为神经网路输入的一部分
    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    # tmp_dornet_true = [Input(shape=(2, 2,))]
    # 这之后y_true这个list中，有三个placeholder，两个yolo不同层的truth，一个预测的truth
    # y_true = y_true + tmp_dornet_true

    model_body = tiny_yolo_and_decision(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        if not weights_path:
            # by_name=True意味着通过layer的名字进行读取
            # by_name=False意味着通过网络结构进行读取
            # 这里应该是因为所有的layer都是默认的名字，所以，直接对应就好
            model_body.load_weights(yolo_weights_path, by_name=True, skip_mismatch=True)
            print('Load weights {}.'.format(yolo_weights_path))
        else:
            model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
            print("Loaded model from {}".format(weights_path))

        # freeze_layer = list(range(len(model_body.layers)))
        # freeze_layer = list(set(freeze_layer).difference(dronet_trainable_layer))
        # try:
        #     for i in freeze_layer:
        #         model_body.layers[i].trainable = False
        # except IndexError as error:
        #     print("There is error when read the layers of network, \
        #             and the reason is %s" % str(error))
        # print('Freeze the {} layers from dronet branch of total {} layers.'
        #       .format(len(dronet_trainable_layer), len(model_body.layers)))

    # Lambda其实就是定义一个任意的层，以上一层为输入，最后得到输出
    # 调用函数使用＊，是将每个元素作为位置参数传入，所以现在相当于传入一个list [*model_body.output, *y_true]，元素数为len(model_body.outpu)+len(y_true)
    # 网络输入因为多了decision的输出，需要改变
    # 这时的*model_body.output是, [y1, y2, steer, coll]，后面会变成只剩yolo_loss
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [model_body.output[0], model_body.output[1], *y_true])
    # 这里是在网络最后又加了一层，所以网络的结果直接就是loss？？？
    model = Model(inputs=[model_body.input, *y_true],
                  outputs=[model_loss, model_body.output[2], model_body.output[3]])
    return model

def data_generator(dronet_train_dataset, annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    # n = len(annotation_lines)
    # i = 0
    while True:
        # image_data = []
        # box_data = []
        # for b in range(batch_size):
        #     if i==0:
        #         np.random.shuffle(annotation_lines)
        #     image, box = get_random_data(annotation_lines[i], input_shape, random=True)
        #     image_data.append(image)
        #     box_data.append(box)
        #     i = (i+1) % n

        # Image transformation is not under thread lock, so it can be done in
        # parallel
        # image_shape = np.array(input_shape + (3,))
        # batch_x = np.zeros((batch_size,) + image_shape, dtype=K.floatx())
        image_shape = input_shape + (3,)
        batch_x = np.zeros((batch_size,) + image_shape, dtype=K.floatx())

        batch_steer = np.zeros((batch_size, 2,), dtype=K.floatx())
        batch_coll = np.zeros((batch_size, 2,), dtype=K.floatx())
        # 作为无用的label
        yolo_default_label = [np.zeros((batch_size, 10, 10, 3, 7), dtype=K.floatx()),
                              np.zeros((batch_size, 20, 20, 3, 7), dtype=K.floatx()),]

        # Build batch of image data
        index_array = dronet_train_dataset._flow_index()
        for i, j in enumerate(index_array):
            fname = dronet_train_dataset.filenames[j]
            x = utils.dronet_load_img(os.path.join(dronet_train_dataset.directory, fname),
                    crop_size=dronet_train_dataset.crop_size,
                    target_size=dronet_train_dataset.target_size)

            x = dronet_train_dataset.image_data_generator.random_transform(x)
            x = dronet_train_dataset.image_data_generator.standardize(x)
            # 存储图片
            batch_x[i] = x

            # Build batch of steering and collision data
            if dronet_train_dataset.exp_type[index_array[i]] == 1:
                # Steering experiment (t=1)
                batch_steer[i,0] =1.0
                batch_steer[i,1] = dronet_train_dataset.ground_truth[index_array[i]]
                batch_coll[i] = np.array([1.0, 0.0])
            else:
                # Collision experiment (t=0)
                batch_steer[i] = np.array([0.0, 0.0])
                batch_coll[i,0] = 0.0
                batch_coll[i,1] = dronet_train_dataset.ground_truth[index_array[i]]

        # batch_y = [batch_steer, batch_coll]
        # batch_y = np.swapaxes(batch_y, 0, 1)
        # batch_y = [yolo_default_label[0], yolo_default_label[1], batch_y]
        # TODO:下面那句话什么作用？？？？？
        # yield之后，其中的元素是否需要使用＊展开需要注意
        # 生成器的返回值结构是（inputs，targets）的tuple
        # 或者（inputs, targets,sample_weight）的tuple
        yield [batch_x, *yolo_default_label], [np.zeros(batch_size), batch_steer, batch_coll]


        # image_data = np.array(image_data)
        # box_data = np.array(box_data)
        # y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        # yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(dronet_train_dataset, annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(dronet_train_dataset, annotation_lines, batch_size, input_shape, anchors, num_classes)

if __name__ == '__main__':
    _main(sys.argv)
