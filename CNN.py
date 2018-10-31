import tensorflow as tf
import numpy as np
import os
import pickle
from PIL import Image
import time
import matplotlib.pyplot as plt
import math

# image size from data set
IMAGE_SIZE = 32

NUM_CHANNELS = 3 # rgb, so 3

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
    '''
    Reads data
    '''

    filenames = [os.path.join('data', 'data_batch_%d' % i)
                 for i in range(1, 6)]

    # takes each batch file and adds it to array after unpickling
    for dir in filenames:
        DATA.append(unpickle(dir))

    print("completed setup")

# File helper
class CifarHelper():

    def __init__(self):
        self.i = 0

        self.all_train_batches = [data_batch1, data_batch2, data_batch3, data_batch4]
        self.test_batch = [test_batch]

        self.training_images = None
        self.training_labels = None

        self.test_images = None
        self.test_labels = None

    def set_up_images(self):
        print("Setting Up Training Images and Labels")

        self.training_images = np.vstack([d[b"data"] for d in self.all_train_batches])
        train_len = len(self.training_images)

        self.training_images = self.training_images.reshape(train_len, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).transpose(0, 2, 3, 1) / 255
        self.training_labels = one_hot_encode(np.hstack([d[b"labels"] for d in self.all_train_batches]), 10)

        print("Setting Up Test Images and Labels")

        self.test_images = np.vstack([d[b"data"] for d in self.test_batch])
        test_len = len(self.test_images)

        self.test_images = self.test_images.reshape(test_len, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).transpose(0, 2, 3, 1) / 255
        self.test_labels = one_hot_encode(np.hstack([d[b"labels"] for d in self.test_batch]), 10)

    def next_batch(self, batch_size):
        x = self.training_images[self.i:self.i + batch_size].reshape(100, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        y = self.training_labels[self.i:self.i + batch_size]
        self.i = (self.i + batch_size) % len(self.training_images)
        return x, y


'''
HELPERS
'''

#
def one_hot_encode(vec, vals=10):
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out


def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)


def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2by2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)


def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b


setup()  # get's images ready

# put DATA into variables
data_batch1 = DATA[0]
data_batch2 = DATA[1]
data_batch3 = DATA[2]
data_batch4 = DATA[3]
test_batch = DATA[4]

# setup images into test and training
ch = CifarHelper()
ch.set_up_images()

# cleanup tf graph
tf.reset_default_graph()

# input variable for session
x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
y_true = tf.placeholder(tf.float32, shape=[None, 10])

# probability a neuron will be dropped
hold_prob = tf.placeholder(tf.float32)

# computes 32 features for each 4 by 4 batch for each image of 3 channels
# input: 3 x 32 x 32 x 100 tensor for the batch
convo_1 = convolutional_layer(x, shape=[4, 4, 3, 32])  # output: 32 x 32 x 32 x 100 tensor for the batch
convo_1_pooling = max_pool_2by2(convo_1)  # output: 32 x 16 x 16 x 100 tensor

convo_2 = convolutional_layer(convo_1_pooling, shape=[4, 4, 32, 64])  # output: 64 x 16 x 16 x 100 tensor
convo_2_pooling = max_pool_2by2(convo_2)  # output: 64 x 8 x 8 x 100 tensor



# convo_3 = convolutional_layer(convo_2_ pooling, shape=[4,4,64,128])
# convo_4 = convolutional_layer(convo_3, shape=[4,4,128,128])
# convo_5 = convolutional_layer(convo_4, shape=[4,4,128,64])
#
#
# convo_flat = tf.reshape(convo_5,[-1,8*8*64])

# flatten image
convo_flat = tf.reshape(convo_2_pooling, [-1, 8 * 8 * 64])  # output: 4096 x 100 tensor

full_layer_one = tf.nn.relu(normal_full_layer(convo_flat, 1024))
full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=hold_prob)

y_pred = normal_full_layer(full_one_dropout, 10)

softmax = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
cross_entropy = tf.reduce_mean(softmax)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()

start_time = time.time()


# shows filters
def showFilter(units, actualy_image, title):
    print(actualy_image.shape)
    filters = units.shape[3]
    plt.figure(1, figsize=(10,10))
    plt.subplots_adjust(hspace=0.8, wspace=0.8)
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")

    plt.subplot(n_rows, n_columns, len(range(filters))+2)
    plt.title('actual image')
    plt.imshow(actualy_image[0], interpolation="nearest", cmap="gray")
    plt.suptitle(title)
    plt.show()

showEachFilter = True

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    steps = 5000
    for i in range(steps):
        batch = ch.next_batch(100)

        sess.run(train, feed_dict={x: batch[0], y_true: batch[1], hold_prob: 0.5})

        if showEachFilter:
            c1 = sess.run(convo_1, feed_dict={x: batch[0], y_true: batch[1], hold_prob: 1})
            showFilter(c1, batch[0], 'Conv Layer 1')

            c2 = sess.run(convo_2, feed_dict={x: batch[0], y_true: batch[1], hold_prob: 1})
            showFilter(c2, batch[0], 'Conv Layer 2')

        # PRINT OUT A MESSAGE EVERY 100 STEPS
        if i % 100 == 0:

            # Test the Train Model
            matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))

            acc = tf.reduce_mean(tf.cast(matches, tf.float32))

            accuracy = sess.run(acc, feed_dict={x: ch.test_images, y_true: ch.test_labels, hold_prob: 1.0})
            print('[{} %] Currently on step {}/{}'.format((i / steps) * 100, i, steps))
            print('Accuracy is: {}\n'.format(accuracy))
            elapsed_time = time.time() - start_time

            print('Time elapsed', elapsed_time, 's')



    elapsed_time = time.time() - start_time

    print('Time it took to run', elapsed_time, 's')

    def predict(fileName):
        dir = "./images/" + fileName
        try:
            img = Image.open(dir)
            img = img.resize([IMAGE_SIZE, IMAGE_SIZE])
            input_image_data = np.array(img)

            value = sess.run(tf.argmax(y_pred, 1),
                             feed_dict={x: [input_image_data], y_true: [list(range(10))], hold_prob: 1.0})
            print("{} : {}".format(fileName, CLASSES[value[0]]))
        except:
            print('file', fileName, 'not found')
            pass

    while(1):
        fileName = input("Enter image name to test with (image should exist in the images folder): ")
        exts = ['.jpg', '.png', '.jpeg']

        if fileName == 'all':
            for file in os.listdir("images"):
                for ext in exts:
                    if file.endswith(ext):
                        predict(file)
        else:
            predict(fileName)

