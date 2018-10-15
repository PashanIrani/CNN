import tensorflow as tf
import os
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

filenames = [os.path.join('data/cifar-10-batches-bin', 'data_batch_%d' % i)
             for i in range(1, 6)]

data = []

# takes each batch file and adds it to array
for dir in filenames:
    data.append(unpickle(dir))










#filename_queue = tf.train.string_input_producer(filenames)

# record bytes are the size of the image,  32 for height and weight, 3 for number of channels (rgb), and 1 for label
#key, value = tf.FixedLengthRecordReader(record_bytes=(32 * 32 * 3) + 1).read(filename_queue)
