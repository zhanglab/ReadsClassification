import os
import ast
import h5py
import keras
import numpy as np
import tensorflow as tf

class generator:
    def __init__(self, file, dataset):
        self.file = file
        self.dataset = dataset

    def __call__(self, *args, **kwargs):
        with h5py.File(self.file, 'r') as hf:
            for array in hf[dataset]:
                yield array

def train_test_loaders(hparams):
    """
    Return one hot labels of the training and testing datasets, and class mapping
    """
    # Load class_mapping
    with h5py.File(self.file, 'r') as hf:
        class_mapping = ast.literal_eval(hf['class_mapping'])

    # Load data
    x_train, y_train = tf.data.Dataset.from_generator(generator(hparams.hdf5, 'train'), tf.int32)
    x_test, y_test = tf.data.Dataset.from_generator(generator(hparams.hdf5, 'val'), tf.int32)

    BUFFER_SIZE = len(x_train)

    def paired():
        z_train = tf.data.Dataset.from_generator(generator(hparams.hdf5, 'training_paired'), tf.int32)
        z_test = tf.data.Dataset.from_generator(generator(hparams.hdf5, 'validation_paired'), tf.int32)

        # Convert label integers into one hot encodings
        z_train = keras.utils.to_categorical(z_train, num_classes=len(hparams.class_mapping))
        z_test = keras.utils.to_categorical(z_test, num_classes=len(hparams.class_mapping))

        # train_dataset = tf.data.Dataset.from_generator(
        #     (())
        # )
        # Slice and batch train and test datasets
        train_dataset = tf.data.Dataset.from_tensor_slices(
            ((x_train, y_train), z_train)).shuffle(BUFFER_SIZE).batch(hparams.global_batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices(
            ((x_test, y_test), z_test)).batch(hparams.global_batch_size)

        return train_dataset, test_dataset, class_mapping

    def unpaired():
        # Convert label integers into one hot encodings
        y_train_new = keras.utils.to_categorical(y_train, num_classes=len(hparams.class_mapping))
        y_test_new = keras.utils.to_categorical(y_test, num_classes=len(hparams.class_mapping))

        # Slice and batch train and test datasets
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (x_train, y_train_new)).shuffle(BUFFER_SIZE).batch(hparams.global_batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices(
            (x_test, y_test_new)).batch(hparams.global_batch_size)

        return train_dataset, test_dataset, class_mapping

    return paired() if hparams.reads == 'paired' else unpaired()