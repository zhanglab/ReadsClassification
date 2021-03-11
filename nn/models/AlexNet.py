from . import model, AbstractNN

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.utils import plot_model

@model('cnn-1D')
class CNN1D(AbstractNN):
    def __init__(self, hparams):
        super(CNN1D, self).__init__(hparams)
        hparams = self.check_hparams(hparams)
        num_classes = len(hparams.class_mapping)
        #if hparams.mode == 'training':
        # Create the model
        self._model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(shape=(hparams.vector_size), dtype='int32'),
                tf.keras.layers.Embedding(input_dim=hparams.num_kmers+1, output_dim=hparams.embedding_size,
                                          input_length=hparams.vector_size, trainable=True),
                tf.keras.layers.Reshape((hparams.vector_size, hparams.embedding_size, 1)),
                tf.keras.layers.Conv2D(96, kernel_size=(11, 11), strides=(4, 4), padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
                tf.keras.layers.Conv2D(256, kernel_size=(5, 5), strides=(1, 1), padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
                tf.keras.layers.Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=4096),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.Dense(units=4096),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.Dense(units=1000),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.Dense(units=num_classes),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('softmax'),
            ]
        )
        
        with open(os.path.join(hparams.output, 'metrics.txt'), 'w+') as f:
            self._model.summary(print_fn=lambda x: f.write(x + '\n'))

    def call(self, strategy):
        self.running_loop(strategy)
