from . import model, AbstractNN

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model

@model('multilstm')
class MultiLSTM(AbstractNN):
    def __init__(self, hparams):
        super(MultiLSTM, self).__init__(hparams)
        hparams = self.check_hparams(hparams)
        num_classes = len(hparams.class_mapping)

        # First LSTM
        input1 = layers.Input(shape=(hparams.vector_size,))
        embed1 = layers.Embedding(input_dim=hparams.num_kmers, output_dim=hparams.embedding_size,
                                  input_length=hparams.vector_size, trainable=True)(input1)
        lstm1 = layers.RNN(layers.LSTMCell(hparams.hidden_size))(embed1)

        # Second LSTM
        input2 = layers.Input(shape=(hparams.vector_size,))
        embed2 = layers.Embedding(input_dim=hparams.num_kmers, output_dim=hparams.embedding_size,
                                  input_length=hparams.vector_size, trainable=True)(input2)
        lstm2 = layers.RNN(layers.LSTMCell(hparams.hidden_size))(embed2)

        # Make the LSTMs parallel
        added = layers.Add()([lstm1, lstm2])
        out = layers.Dense(num_classes, activation='softmax')(added)

        # Create the model
        self._model = models.Model(inputs=[input1, input2], outputs=out)

        plot_model(self._model, to_file=os.path.join(hparams.output, 'multilstm-model.png'),
                   show_shapes=True, show_layer_names=True)

    def call(self, strategy):
        self.running_loop(strategy)