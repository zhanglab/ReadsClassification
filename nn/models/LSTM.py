import os
import sys
import tensorflow as tf
import argparse
from sklearn import metrics

from .train import distributed_train_epoch
from .test import distributed_test_epoch
from ..loader import train_test_loaders
from ..summarize import LearningCurvesPlot

class AbstractLSTM(tf.keras.Model):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = self.check_hparams(hparams)
        self.optimizer = tf.keras.optimizers.Adamax(learning_rate=0.01)
        self.checkpoint_path = os.path.join(
            self.hparams.output, 'LSTM-{0}-model-'.format(self.hparams.model))
        self.loss_object = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
        self.test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')

        self.val_loss_before = -1
        self.patience = 0
        self.stop_training = False

        self.true_classes = list()
        self.predicted_classes = list()
        self.probability_threshold = 0.5
        self.probabilities = list()
        self.unclassified = int()

    @staticmethod
    def check_hparams(hparams):
        if isinstance(hparams, dict):
            return argparse.Namespace(**hparams)
        return hparams

    def set_dataset(self, strategy):
        """
        Process the dataset
        """
        self.hparams.global_batch_size = (self.hparams.batch_size * strategy.num_replicas_in_sync)
        train_dataset, test_dataset = train_test_loaders(self.hparams)

        # Distribute train and test datasets
        train_dataset = strategy.experimental_distribute_dataset(train_dataset)
        test_dataset = strategy.experimental_distribute_dataset(test_dataset)

        self.loaders = {'train': train_dataset, 'test': test_dataset}

    def check_loaders(self, strategy):
        """
        Load the dataset if it has not already been loaded
        """
        if not hasattr(self, 'loaders'):
            self.set_dataset(strategy)

    def decay(self, epoch):
        if epoch >= 100:
            return 0.00001
        elif epoch >= 50:
            return 0.0001
        return 0.01

    def running_loop(self, strategy):
        train_epoch_loss = []
        train_epoch_accuracy = []
        test_epoch_loss = []
        test_epoch_accuracy = []

        self.check_loaders(strategy)

        with open(os.path.join(self.hparams.output, 'epochs.txt'), 'a+') as f:
            for epoch in range(self.hparams.epochs):
                self.optimizer.learning_rate = self.decay(epoch)
                f.write('Learning rate at epoch {0}: {1}'.format(epoch, self.optimizer.learning_rate))

                # Train
                train_total_loss, num_train_batches = distributed_train_epoch(self, strategy)

                # Insert train loss and train accuracy after one epoch in corresponding lists
                train_epoch_loss.append(train_total_loss / num_train_batches)
                train_epoch_accuracy.append(self.train_accuracy.result())

                # use validation set to get performance
                distributed_test_epoch(self, strategy)
                # insert test loss and test accuracy after one epoch in corresponding lists
                test_epoch_accuracy.append(self.test_accuracy.result())
                test_epoch_loss.append(self.test_loss.result())

                template1 = ('Epoch: {}, Train Loss: {}, Train Accuracy: {}, Test Loss: {}, Test Accuracy: {}')
                template2 = ('Epoch: {}, Total Train Loss: {}, Train batches: {}, Train Accuracy: {}, '
                             'Test Loss: {}, Test Accuracy: {}')
                f.write(template1.format(epoch + 1, train_total_loss / num_train_batches,
                                       self.train_accuracy.result() * 100,
                                       self.test_loss.result(),
                                       self.test_accuracy.result() * 100))
                f.write(template2.format(epoch + 1, train_total_loss, num_train_batches,
                                       self.train_accuracy.result() * 100,
                                       self.test_loss.result(),
                                       self.test_accuracy.result() * 100))

                val_loss = self.test_loss.result()

                if val_loss == self.val_loss_before:
                    self.patience += 1
                    if self.patience == 3:
                        self.stop_training = True
                else:
                    self.patience = 0

                self.val_loss_before = val_loss

                # Get filtered results, learning curves, ROC and recall precision curves after last epoch
                if epoch == self.hparams.epochs - 1 or self.stop_training == True:
                    with open(os.path.join(self.hparams.output, 'metrics.txt'), 'a+') as out:
                        # Print report on precision, recall, f1-score
                        out.write(metrics.classification_report(self.true_classes, self.predicted_classes, digits=3, zero_division=0))
                        out.write('\nConfusion matrix:\n {}'.format(
                            metrics.confusion_matrix(self.true_classes, self.predicted_classes)))

                    # Plot precision-recall curves
                    metrics.classification_report(self.true_classes, self.predicted_classes, zero_division=0, output_dict=True)

                    # GetFilteredResults(self)
                    LearningCurvesPlot(self, train_epoch_loss, train_epoch_accuracy, test_epoch_loss, test_epoch_accuracy, epoch + 1)
                    self._model.save_weights(
                        self.checkpoint_path + 'V2epoch-{0}-batch-{1}.ckpt'.format(epoch + 10, num_train_batches))
                    break

                # Resets all of the metric (train + test accuracies + test loss) state variables.
                self.train_accuracy.reset_states()
                self.test_accuracy.reset_states()
                self.test_loss.reset_states()
                self.true_classes = list()
                self.predicted_classes = list()
                self.probabilities = list()
                self.unclassified = int()
                self.val_loss = test_epoch_loss