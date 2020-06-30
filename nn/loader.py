import os
import json
import keras
import numpy as np
import tensorflow as tf

def read_dataset(input_folder):
    with open(os.path.join(input_folder, 'class_mapping.json')) as f:
        class_mapping = json.load(f)
    return class_mapping

def train_test_loaders(hparams):
    """
    Return one hot labels of the training and testing datasets
    """
    # Load data
    traindata = np.load(os.path.join(hparams.input, 'train_data_{0}.npy'.format(hparams.model)), allow_pickle=True)
    valdata = np.load(os.path.join(hparams.input, 'val_data_{0}.npy'.format(hparams.model)), allow_pickle=True)

    def bidirectional():
        x1_train = np.array([i[0] for i in traindata])
        x2_train = np.array([i[1] for i in traindata])
        y_train = np.array([i[2] for i in traindata])
        x1_test = np.array([i[0] for i in valdata])
        x2_test = np.array([i[1] for i in valdata])
        y_test = np.array([i[2] for i in valdata])

        BUFFER_SIZE = len(x1_train)

        # Convert label integers into one hot encodings
        y_train = keras.utils.to_categorical(y_train, num_classes=len(hparams.class_mapping))
        y_test = keras.utils.to_categorical(y_test, num_classes=len(hparams.class_mapping))

        # Slice and batch train and test datasets
        train_dataset = tf.data.Dataset.from_tensor_slices(((x1_train, x2_train), y_train)).shuffle(
            BUFFER_SIZE).batch(hparams.global_batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices(((x1_test, x2_test), y_test)).batch(hparams.global_batch_size)
        return train_dataset, test_dataset

    def unidirectional():
        x_train = np.array([i[0] for i in traindata])
        y_train = np.array([i[1] for i in traindata])
        x_test = np.array([i[0] for i in valdata])
        y_test = np.array([i[1] for i in valdata])

        BUFFER_SIZE = len(x_train)

        # Convert label integers into one hot encodings
        y_train = keras.utils.to_categorical(y_train, num_classes=len(hparams.class_mapping))
        y_test = keras.utils.to_categorical(y_test, num_classes=len(hparams.class_mapping))

        # Slice and batch train and test datasets
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (x_train, y_train)).shuffle(BUFFER_SIZE).batch(hparams.global_batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices(
            (x_test, y_test)).batch(hparams.global_batch_size)
        return train_dataset, test_dataset

    return bidirectional() if hparams.model == 'bidirectional' else unidirectional()