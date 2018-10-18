import tensorflow as tf
import numpy as np
import os
import pickle

IMAGE_SIZE = 32
IMAGE_SIZE_CROPPED = 24
NUM_CHANNELS = 3
NUM_CLASSES = 10
CLASSES = ['airplane',
 'automobile',
 'bird',
 'cat',
 'deer',
 'dog',
 'frog',
 'horse',
 'ship',
 'truck']
DATA = []  # holds images

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def setup():
    filenames = [os.path.join('data', 'data_batch_%d' % i)
                 for i in range(1, 6)]

    # takes each batch file and adds it to array
    for dir in filenames:
        DATA.append(unpickle(dir))

    print("completed setup")


def train():
    tf.reset_default_graph()
    data = DATA
    x = DATA[0][b'data'].astype(np.float32)
    l = DATA[0][b'labels']
    imageNum = 566

    p = network(x, l, training=True)
    print(p)

    sess = tf.Session()

    sess.run()
    # print(loss)


def network(images, labels, training=False):

    input_layer = tf.reshape(images, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=training == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    print(logits)

    # return y_pred, loss



setup()
train()

