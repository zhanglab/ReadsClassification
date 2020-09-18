import os
import ast
import h5py
import numpy as np
import tensorflow as tf
from glob import glob

# make number of classes automated
def read_tfrecord(proto_example):
    data_description = {
        'kmers': tf.io.FixedLenFeature((141,), tf.int64),
        'label': tf.io.FixedLenFeature((103,), tf.float32)
    }
    # load one example
    example = tf.io.parse_single_example(proto_example, data_description)
    return example['kmers'], example['label']

def train_test_loaders(hparams, set_type):
    """
    Return reads and one hot labels of the training or testing datasets
    """
    # Get list of TFRecord files
    tfrecord_path = os.path.join(hparams.input + '/tfrecords/{}'.format(set_type), "*")
    tfrecord_files = sorted(glob(tfrecord_path), key=os.path.getctime)
    # Load data as shards
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(map_func=read_tfrecord, num_parallel_calls=16)
    dataset = dataset.batch(batch_size=hparams.global_batch_size)
    dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
