import os
import json
import keras
import numpy as np

def read_dataset(input_folder):
    with open(os.path.join(input_folder, 'class_mapping.json')) as f:
        class_mapping = json.load(f)
    return class_mapping

def train_test_loaders(args):
    """
    Return one hot labels of the training and testing datasets
    """
    # Load data
    traindata = np.load(os.path.join(args.input, 'train_data_{0}.npy'.format(args.model)), allow_pickle=True)
    valdata = np.load(os.path.join(args.input, 'val_data_{0}.npy'.format(args.model)), allow_pickle=True)

    x_train = np.array([i[0] for i in traindata])
    y_train = np.array([i[1] for i in traindata])
    x_test = np.array([i[0] for i in valdata])
    y_test = np.array([i[1] for i in valdata])

    # Convert label integers into one hot encodings
    y_train = keras.utils.to_categorical(y_train, num_classes=len(args.class_mapping))
    y_test = keras.utils.to_categorical(y_test, num_classes=len(args.class_mapping))
    return x_train, y_train, x_test, y_test