from . import model, AbstractNN

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model

@model('multilstm-3')
class MultiLSTM3(AbstractNN):
    def __init__(self, hparams):
        super(MultiLSTM3, self).__init__(hparams)
        hparams = self.check_hparams(hparams)
        num_classes = len(hparams.class_mapping)

        # Ensure that the reshape layer is the same size as the LSTM layers
        assert (hparams.embedding_size * hparams.vector_size) == hparams.hidden_size

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
        reshaped = layers.Reshape((hparams.vector_size, hparams.embedding_size))(added)

        # Feed the LSTM layers into a third LSTM
        lstm3 = layers.RNN(layers.LSTMCell(hparams.hidden_size))(reshaped)
        out = layers.Dense(num_classes, activation='softmax')(lstm3)

        # Create the model
        self._model = models.Model(inputs=[input1, input2], outputs=out)

        plot_model(self._model, to_file=os.path.join(hparams.output, 'multilstm-3-model.png'),
                   show_shapes=True, show_layer_names=True)

    def call(self, strategy):
        self.running_loop(strategy)