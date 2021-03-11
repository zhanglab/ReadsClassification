import os
import numpy as np
import tensorflow as tf


def read_tfrecord_training(proto_example):
    print('in read_tfrecord', tf.executing_eagerly())
    data_description = {
        'read': tf.io.VarLenFeature(tf.int64),
        'label': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
    }
    # load one example
    parsed_example = tf.io.parse_single_example(serialized=proto_example, features=data_description)
    read = parsed_example['read']
    label = tf.cast(parsed_example['label'], tf.int64)
    label = tf.one_hot(label, 160)
    read = tf.sparse.to_dense(read)
    return read, label[0]

def read_tfrecord_testing(proto_example):
    print('in read_tfrecord', tf.executing_eagerly())
    data_description = {
        'read': tf.io.VarLenFeature(tf.int64),
        'label': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
    }
    # load one example
    parsed_example = tf.io.parse_single_example(serialized=proto_example, features=data_description)
    read = parsed_example['read']
    label = tf.cast(parsed_example['label'], tf.int64)
    read = tf.sparse.to_dense(read)
    return read, label

def input_training_testing(hparams, set_type):
    """
    Return reads and one hot labels for training and testing sets
    """
    # Get list of TFRecord files
    tfrecord_path = os.path.join(hparams.input + '/tfrecords/{}'.format(set_type), "*")
    # Load data as shards
    dataset = tf.data.Dataset.list_files(tfrecord_path)
    dataset = dataset.interleave(lambda x: tf.data.TFRecordDataset(x), num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                 deterministic=False)
    dict_functions = {'testing': read_tfrecord_testing, 'training': read_tfrecord_training, 'evaluation': read_tfrecord_training}
    dict_vector = {'testing': None, 'training': len(hparams.class_mapping), 'evaluation': len(hparams.class_mapping)}
    dataset = dataset.map(map_func=dict_functions[hparams.mode], num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.padded_batch(hparams.global_batch_size,
                                   padded_shapes=(tf.TensorShape([hparams.vector_size]), tf.TensorShape([dict_vector[hparams.mode]])),)
    dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def read_tfrecord_inference(proto_example):
    print('in read_tfrecord', tf.executing_eagerly())
    data_description = {
        'read': tf.io.VarLenFeature(tf.int64),
        'read_id': tf.io.VarLenFeature(tf.string)
    }
    # load one example
    parsed_example = tf.io.parse_single_example(serialized=proto_example, features=data_description)
    read = parsed_example['read']
    read = tf.sparse.to_dense(read)
    read_id = parsed_example['read_id']
    # convert sparse tensor to dense tensor
    dense_read_id = tf.sparse.to_dense(read_id, default_value=b'')
    return read, dense_read_id

def input_inference(hparams, set_type):
    """
    Return reads and read ids for inference
    """
    print('in train_test_loaders', tf.executing_eagerly())
    # Get list of TFRecord files
    tfrecord_path = os.path.join(hparams.input + '/tfrecords/{}'.format(set_type), "*")
    # Load data as shards
    dataset = tf.data.Dataset.list_files(tfrecord_path)
    dataset = dataset.interleave(lambda x: tf.data.TFRecordDataset(x), num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                 deterministic=False)
    dataset = dataset.map(map_func=read_tfrecord_inference, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.padded_batch(hparams.global_batch_size,
                                   padded_shapes=(tf.TensorShape([hparams.vector_size]), tf.TensorShape([None, ])))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset



