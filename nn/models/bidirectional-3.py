from . import model, AbstractNN

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model

@model('bidirectional-3')
class Bidirectional3(AbstractNN):
    def __init__(self, hparams):
        super(Bidirectional3, self).__init__(hparams)
        hparams = self.check_hparams(hparams)
        num_classes = len(hparams.class_mapping)

        # Ensure that the reshape layer is the same size as the LSTM layers
        assert (hparams.embedding_size * hparams.vector_size) == hparams.hidden_size

        # First bidirectional LSTM
        input1 = layers.Input(shape=(hparams.vector_size,))
        embed1 = layers.Embedding(input_dim=hparams.num_kmers, output_dim=hparams.embedding_size,
                                  input_length=hparams.vector_size, trainable=True)(input1)
        lstm1 = layers.Bidirectional(layers.LSTM(hparams.hidden_size))(embed1)

        # Second bidirectional LSTM
        input2 = layers.Input(shape=(hparams.vector_size,))
        embed2 = layers.Embedding(input_dim=hparams.num_kmers, output_dim=hparams.embedding_size,
                                  input_length=hparams.vector_size, trainable=True)(input2)
        lstm2 = layers.Bidirectional(layers.LSTM(hparams.hidden_size))(embed2)

        # Make the LSTMs parallel
        added = layers.Add()([lstm1, lstm2])
        reshaped = layers.Reshape((hparams.vector_size, hparams.embedding_size))(added)

        # Feed the LSTM layers into a third bidirectional LSTM
        lstm3 = layers.Bidirectional(layers.LSTM(hparams.hidden_size))(reshaped)
        out = layers.Dense(num_classes, activation='softmax')(lstm3)

        # Create the model
        self._model = models.Model(inputs=[input1, input2], outputs=out)

        # Output model summary to a file
        with open(os.path.join(hparams.output, 'metrics.txt'), 'w+') as f:
            self._model.summary(print_fn=lambda x: f.write(x + '\n'))

        plot_model(self._model, to_file=os.path.join(hparams.output, 'bidirectional-3-model.png'),
                   show_shapes=True, show_layer_names=True)

    def call(self, strategy):
        self.running_loop(strategy)