import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from sklearn import metrics

from .train import distributed_train_epoch
from .test import distributed_test_epoch
from ..loader import train_test_loaders
from ..summarize import LearningCurvesPlot

class AbstractNN(tf.keras.Model):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = self.check_hparams(hparams)
        self.optimizer = tf.keras.optimizers.Adamax(learning_rate=self.hparams.learning_rate)
        self.checkpoint_path = os.path.join(
            self.hparams.output, 'LSTM-{0}-model-'.format(self.hparams.model))
        self.loss_object = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
        self.test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')

        self.val_loss_before = -1
        self.lowest_val_loss = 1
        self.patience = 0
        self.wait = 0
        self.best_weights = None
        self.stopped_epoch = 0
        self.best = np.Inf
        self.stop_training = False
        self.min_true_class = 0
        self.min_pred_class = 0
        self.min_epoch = 0
        self.found_min = False

        self.true_classes = list()
        self.predicted_classes = list()

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
        train_dataset, test_dataset, class_mapping = train_test_loaders(self.hparams)

        # Distribute train and test datasets
        train_dataset = strategy.experimental_distribute_dataset(train_dataset)
        test_dataset = strategy.experimental_distribute_dataset(test_dataset)

        self.loaders = {'class_mapping': class_mapping,'train': train_dataset, 'test': test_dataset}

    def check_loaders(self, strategy):
        """
        Load the dataset if it has not already been loaded
        """
        if not hasattr(self, 'loaders'):
            self.set_dataset(strategy)

    def running_loop(self, strategy):
        train_epoch_loss = []
        train_epoch_accuracy = []
        test_epoch_loss = []
        test_epoch_accuracy = []

        self.check_loaders(strategy)

        with open(os.path.join(self.hparams.output, 'epochs.txt'), 'a+') as f:
            for epoch in range(self.hparams.epochs):
                f.write('Learning rate at epoch {0}: {1}'.format(epoch, self.optimizer.learning_rate))

                # Train
                train_total_loss, num_train_batches = distributed_train_epoch(self, strategy)
                train_loss = train_total_loss / num_train_batches
                # Insert train loss and train accuracy after one epoch in corresponding lists
                train_epoch_loss.append(train_loss)
                train_epoch_accuracy.append(self.train_accuracy.result())

                # use validation set to get performance
                distributed_test_epoch(self, strategy)
                # insert test loss and test accuracy after one epoch in corresponding lists
                test_epoch_accuracy.append(self.test_accuracy.result())
                test_epoch_loss.append(self.test_loss.result())

                template1 = ('Epoch: {}, Train Loss: {}, Total Train Loss: {}, Train batches: {}, Train Accuracy: {}, '
                             '\nTest Loss: {}, Test Accuracy: {}\n')
                f.write(template1.format(epoch + 1, train_loss,
                                         train_total_loss, num_train_batches,
                                         self.train_accuracy.result() * 100,
                                         self.test_loss.result(),
                                         self.test_accuracy.result() * 100))

                self.on_epoch_end(epoch)

                # Get filtered results, learning curves, ROC and recall precision curves after last epoch
                if epoch == self.hparams.epochs - 1 or self.stop_training:
                    with open(os.path.join(self.hparams.output, 'metrics.txt'), 'a+') as out:
                        # Print report on precision, recall, f1-score
                        out.write(metrics.classification_report(self.true_classes, self.predicted_classes,
                                                                target_names=self.loaders.class_mapping.values(),
                                                                digits=3, zero_division=0))
                        out.write('\nConfusion matrix:\n {}\n'.format(
                            metrics.confusion_matrix(self.true_classes, self.predicted_classes)))

                    # GetFilteredResults(self)
                    LearningCurvesPlot(self, train_epoch_loss, train_epoch_accuracy,
                                       test_epoch_loss, test_epoch_accuracy, epoch + 1)
                    self._model.save_weights(self.checkpoint_path +
                                                          'V2epoch-{0}-batch-{1}.ckpt'.format(epoch + 10, num_train_batches))
                    if self.found_min:
                        self._model.set_weights(self.best_weights)
                        self._model.save_weights(self.checkpoint_path + 'minloss.ckpt')
                        with open(os.path.join(self.hparams.output, 'metrics.txt'), 'a+') as out:
                            out.write('\nLowest validation loss epoch: {}\n'.format(self.min_epoch + 1))
                            out.write('Test loss: {}\tTest accuracy: {}\n'.format(
                                test_epoch_loss[self.min_epoch], test_epoch_accuracy[self.min_epoch]))
                            # Print report on precision, recall, f1-score of lowest validation loss
                            out.write(metrics.classification_report(self.min_true_class, self.min_pred_class,
                                                                    target_names=self.loaders.class_mapping.values(),
                                                                    digits=3, zero_division=0))
                            out.write('\nConfusion matrix for lowest validation loss:\n {}'.format(
                                metrics.confusion_matrix(self.min_true_class, self.min_pred_class)))
                    break

                # Resets all of the metric (train + test accuracies + test loss) state variables.
                self.train_accuracy.reset_states()
                self.test_accuracy.reset_states()
                self.test_loss.reset_states()
                self.true_classes = list()
                self.predicted_classes = list()
                self.val_loss = test_epoch_loss

    # Custom callback implementation
    def on_epoch_end(self, epoch):
        # Early stopping
        val_loss = self.test_loss.result()
        if self.wait < 5:
            self.model_checkpoint(val_loss, epoch)
        else:
            self.found_min = True
            self.early_stopping(val_loss)

    def model_checkpoint(self, val_loss, epoch):
        # ModelCheckpoint
        if val_loss < self.best:
            self.best = val_loss
            self.lowest_val_loss = val_loss
            self.best_weights = self._model.get_weights()
            self.wait = 0
            self.min_true_class = self.true_classes
            self.min_pred_class = self.predicted_classes
            self.min_epoch = epoch
        else:
            self.wait += 1

    def early_stopping(self, val_loss):
        # Calculate percent difference
        if abs(100 * (val_loss - self.val_loss_before) / self.val_loss_before) < 5:
            self.patience += 1
            if self.patience == 5:
                if self.optimizer.learning_rate == 0.0001:
                    self.optimizer.learning_rate = 0.00001
                    self.patience = 0
                else:
                    self.stop_training = True
        else:
            self.patience = 0

        self.val_loss_before = val_loss