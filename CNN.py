import tensorflow as tf
import os
import pickle

IMAGE_SIZE = 32

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

DATA = [] # holds images

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def setup():

    filenames = [os.path.join('data/cifar-10-batches-bin', 'data_batch_%d' % i)
                 for i in range(1, 6)]

    # takes each batch file and adds it to array
    for dir in filenames:
        DATA.append(unpickle(dir))


def train():
    # todod
    print('train')
    print(DATA[0][b'data'][0])
    image = tf.convert_to_tensor(DATA[0][b'data'][0], dtype=tf.float64)
    input = [image, IMAGE_SIZE, IMAGE_SIZE, 3]
    tf.nn.conv2d(input, [IMAGE_SIZE, IMAGE_SIZE, 3, 1], strides=[1, 1, 1, 1],
                     padding='SAME')
    print('lol')



setup()
train()








#filename_queue = tf.train.string_input_producer(filenames)

# record bytes are the size of the image,  32 for height and weight, 3 for number of channels (rgb), and 1 for label
#key, value = tf.FixedLengthRecordReader(record_bytes=(32 * 32 * 3) + 1).read(filename_queue)
