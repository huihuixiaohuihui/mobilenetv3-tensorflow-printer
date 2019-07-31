# coding=utf-8
# 兼容python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing as mt

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Tuple
import os

# 分类问题，总共有250个类
NUM_CLASSES = 3
# 训练批次大小
TRAIN_BATCH_SIZE = 1
# 图像每个像素的每个通道的最大值，对于8位图像，就是255
IMAGE_DEPTH = 128
# 包含训练集的文本
TRAIN_LIST = 'data/data.txt'

def build_dataset(
    shape: Tuple[int, int],
    name: str="mnist",
    train_batch_size: int=32,
    valid_batch_size: int=32,
    ):
    dataSet = {}
    dataSet['num_classes'] = 3
    dataSet['channels'] = 3

    dataSet['train'], dataSet['train_steps_per_epoch'] = prepare_dataset(
        path='data/data_train.txt',
        num_classes = dataSet['num_classes'],
        train_batch_size = train_batch_size,
        shape=shape,
    )
    dataSet['test'], dataSet['val_steps_per_epoch'] = prepare_dataset(
        path='data/data_val.txt',
        num_classes = dataSet['num_classes'],
        train_batch_size = valid_batch_size,
        shape=shape,
    )

    return dataSet


def prepare_dataset(num_classes, train_batch_size, shape, path=""):
    """
    prepaer dataset using tf.data.Dataset
    :param path: the list file like data/train_lists_demo.txt
    and data/val_lists_demo.txt
    :return: a Dataset object
    """

    # read image list files name and labels
    lists_and_labels = np.loadtxt(path, dtype=str).tolist()
    # shuffle dataset
    np.random.shuffle(lists_and_labels)
    # split lists an labels
    list_files, labels = zip(*[(l[0], int(l[1])) for l in lists_and_labels])
    # one_shot encoding on labels
    one_shot_labels = keras.utils.to_categorical(labels, num_classes).astype(dtype=np.int32)
    # make data set
    dataset = tf.data.Dataset.from_tensor_slices((tf.constant(list_files), tf.constant(one_shot_labels)))
    # perform function parse_image on each pair of (data, label)
    # dataset = dataset.map(lambda data : _parse_function(data,shape,num_classes,3))

    dataset = dataset.map(_parse_image, num_parallel_calls=mt.cpu_count())
    # set the batch size, Very important function!
    dataset = dataset.batch(train_batch_size)
    # repeat forever
    dataset = dataset.repeat()

    # compute steps_per_epoch
    steps_per_epoch = np.ceil(len(labels) / train_batch_size).astype(np.int32)
    # print(dataset.num_examples)

    # dataSet['train'] = dataset
    # dataSet['num_train'] = len(labels)
    return dataset, steps_per_epoch
    # return dataSet


def _parse_image(filename, label):
    """
    read and pre-process image
    """
    image_string = tf.read_file(filename)
    image_decoded = tf.io.decode_jpeg(image_string,channels=3)
    # must convert dtype here !!!
    image_converted = tf.cast(image_decoded, tf.float32)
    image_scaled = tf.image.resize_images(image_converted, (224,224), method=0)
    image = tf.reshape(image_scaled, (224,224, 3))
    image = tf.divide(tf.subtract(image, 255 / 2), 255)
    # image = tf.expand_dims(image, 0)
    # label = tf.one_hot(label, depth=3)
    return image, label

def _parse_function(data, shape, num_classes, channels):
    height, width = shape
    image = data["image"]
    label = data["label"]

    image = tf.cast(image, dtype=tf.float32)
    image = tf.image.resize_images(image, (height,width))
    image = tf.reshape(image, (height,width, channels))
    image = image / 255.0
    image = image - 0.5
    image = image * 2.0
    label = tf.one_hot(label, depth=num_classes)

    return image, label

def read_image(filename):
    """
    read image defined by filename
    :param filename: the path of a image
    :return: a numpy array
    """
    with tf.Session() as sess:
        image, label = _parse_image(tf.constant(filename, dtype=tf.string), tf.constant(0))
        # writer = tf.summary.FileWriter('logs')
        # summary_op = tf.summary.image("image1", image)
        image= sess.run(image)
        # summary = sess.run(summary_op)
        # writer.add_summary(summary)
        
        # print(image, label)
        print(image.shape)
        return image


def inputs_test():
    """
    test function prepare_dataset
    """
    dataset, steps = prepare_dataset(3, 1, [128,128], path=TRAIN_LIST)
    print('shapes:', dataset.output_shapes)
    print('types:', dataset.output_types)
    print('steps:', steps)
    data_it = dataset.make_one_shot_iterator()
    next_data = data_it.get_next()

    with tf.Session() as sess:
        for i in range(10):
            data, label = sess.run(next_data)
            print(len(data), len(label), data.shape, label.shape, np.min(data), np.max(data)) 

if __name__ == '__main__':
    # inputs_test()
    # read_image('data/image_shipping_0709/0000001.jpg')
    # read_image('data/special_handling_images/0200001.jpg')
    # read_image('data/nutrition_facts_images/0100333.jpg')
    path = 'data/nutrition_facts_images'
    # filelist = path
    filelist = os.listdir(path)
    total_num = len(filelist)
    for item in filelist:
        print(path+'/'+item)
        file_name = path+'/'+item
        read_image(file_name)