import tensorflow as tf
import os

IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

data_dir = 'data/cifar-10-batches-bin'
filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
             for i in range(1, 6)]

filename_queue = tf.train.string_input_producer(filenames)

# record bytes are the size of the image,  32 for height and weight, 3 for number of channels (rgb), and 1 for label
key, value = tf.FixedLengthRecordReader(record_bytes=(32 * 32 * 3) + 1).read(filename_queue)
