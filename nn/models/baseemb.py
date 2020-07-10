from . import model, AbstractLSTM

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model

@model('base-emb')
class BaseEmb(AbstractLSTM):
    def __init__(self, hparams):
        super(BaseEmb, self).__init__(hparams)
        hparams = self.check_hparams(hparams)
        num_classes = len(hparams.class_mapping)
        # Create the model
        self._model = keras.Sequential(
            [
                layers.Embedding(input_dim=4, output_dim=hparams.embedding_size,
                                                input_length=hparams.length, trainable=True),
                layers.Dropout(hparams.dropout_rate),
                layers.RNN(tf.keras.layers.LSTMCell(hparams.hidden_size)),
                layers.Dense(num_classes, activation='softmax')
            ]
        )

        plot_model(self._model, to_file=os.path.join(hparams.output, 'baseemb-model.png'),
                   show_shapes=True, show_layer_names=True)

    def call(self, strategy):
        self.running_loop(strategy)
