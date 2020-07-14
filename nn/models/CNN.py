from . import model, AbstractLSTM

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model

@model('cnn')
class CNN(AbstractLSTM):
    def __init__(self, hparams):
        super(CNN, self).__init__(hparams)
        hparams = self.check_hparams(hparams)
        num_classes = len(hparams.class_mapping)

        # Create the model
        self._model = keras.Sequential()
        self._model.add(layers.Input(shape=(hparams.vector_size),), dtype='int32')
        self._model.add(layers.Embedding(input_dim=hparams.num_kmers, output_dim=hparams.embedding_size,
                                   input_length=hparams.vector_size, trainable=True))
        self._model.add(layers.Reshape((hparams.vector_size, hparams.embedding_size, 1)))
        self._model.add(layers.Conv2D(hparams.filter, kernel_size=(hparams.kernel, hparams.embedding_size),
                                activation='relu', kernel_regularizers.l2(3)))
        self._model.add(layers.MaxPool2D(hparams.vector_size - hparams.kernel + 1, 1), strides=(1,1))
        self._model.add(layers.Flatten())
        self._model.add(layers.Dense(units=num_classes, activation='softmax'))

        plot_model(self._model, to_file=os.path.join(hparams.output, 'cnn-model.png'),
                   show_shapes=True, show_layer_names=True)

    def call(self, strategy):
        self.running_loop(strategy)
