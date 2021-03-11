import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from sklearn import metrics
from tensorflow.keras import mixed_precision
import datetime
import json
import math

from .train import train_step
from .train import distributed_train_epoch
from .test import distributed_test_epoch
from ..loader import input_training_testing, input_inference
from ..summarize import LearningCurvesPlot, ROCcurve, RecallPrecisionCurves
from ..utils import kmer_dictionary

class AbstractNN(tf.keras.Model):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = self.check_hparams(hparams)
        self.optimizer = tf.keras.optimizers.Adamax(learning_rate=self.hparams.learning_rate)
        # use loss scaling: multiply the loss by some large number (called the loss scale)
        # this will cause the gradients to scale to this large number and reduce
        # the chance of underflow
        # use tf.keras.mixed_precision.LossScaleOptimizer to dynamically determine the loss scale
        #self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer)
        self.checkpoint_path = os.path.join(
            self.hparams.output, 'ckpts', '{0}-model-'.format(self.hparams.model))
        self.loss_object = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy_species')
        self.test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy_species')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss_species')
        self.num_epochs_done = 0
        self.num_copy = 1

        self.val_loss_before = -1
        self.lowest_val_loss = 1
        self.patience = 0
        self.wait = 0
        self.best_weights = None
        self.stopped_epoch = 0
        self.best = np.Inf
        self.stop_training = False
        self.found_min = False
        self.min_true_class = 0
        self.min_pred_class = 0
        self.min_epoch = 0
        
        self.true_classes = list()
        self.predicted_classes = list()
        self.predictions = list()
        self.one_hot = list()
        self.read_ids = list()


    @staticmethod
    def check_hparams(hparams):
        if isinstance(hparams, dict):
            return argparse.Namespace(**hparams)
        return hparams

    def set_dataset(self, strategy, set_type):
        """
        Process the dataset
        """
        if self.hparams.mode == 'training' or self.hparams.mode == 'testing':
            dataset = input_training_testing(self.hparams, set_type)
            # Distribute train and test datasets
            dataset_distr = strategy.experimental_distribute_dataset(dataset)
            return dataset_distr
        elif self.hparams.mode == 'inference':
            dataset = input_inference(self.hparams, set_type)
            # Distribute train and test datasets
            dataset_distr = strategy.experimental_distribute_dataset(dataset)
            return dataset_distr

    def check_loaders(self, strategy):
        """
        Load the dataset if it has not already been loaded
        """
        if not hasattr(self, 'loaders'):
            self.hparams.global_batch_size = (self.hparams.batch_size * strategy.num_replicas_in_sync)
            if self.hparams.mode == 'training':
                train_set = self.set_dataset(strategy, 'train')
                test_set = self.set_dataset(strategy, 'test')
                self.loaders = {'train': train_set, 'test': test_set}
            elif self.hparams.mode == 'inference' or self.hparams.mode == 'testing':
                test_set = self.set_dataset(strategy, 'test')
                self.loaders = {'test': test_set}
            print('LOADED DATASET')

    def adjust_predictions(self, predictions):
        # adjust vectors of predictions for the desired rank
        new_predictions = []
        for i in range(len(self.hparams.integer_mapping)):
            taxon_pred = np.sum(predictions[self.hparams.rank_mapping_dict[self.hparams.integer_mapping[i]]])
            new_predictions.append(taxon_pred)
        predicted_class = np.argmax(new_predictions)
        return predicted_class, new_predictions[predicted_class]

    def running_loop(self, strategy):
        # create lists to store loss and accuracy
        train_epoch_loss = []
        train_epoch_accuracy = []
        test_epoch_loss = []
        test_epoch_accuracy = []

        # load model is resuming training or in testing/inference modes
        if self.hparams.checkpoint is not None:
            # load model
            self._model = tf.keras.models.load_model(self.hparams.checkpoint)
            #print(f'model summary: {self._model.summary()}')
            print('MODEL LOADED')

        # load data
        self.check_loaders(strategy)


        with open(os.path.join(self.hparams.output, f'{self.hparams.mode}-output.txt'), 'a+') as f:
            if self.hparams.mode == 'training':
#                tf.profiler.experimental.start(self.hparams.profile_log_dir)
                for epoch in range(self.hparams.num_epochs_done, self.hparams.epochs, 1):
                    with self.hparams.learning_rate_summary_writer.as_default():
                        tf.summary.scalar('learning_rate', self.optimizer.learning_rate, step=epoch)
                        self.hparams.learning_rate_summary_writer.flush()
                    train_total_loss, num_train_batches = distributed_train_epoch(self, strategy)
                    print(f'Number of train batches: {num_train_batches}')
                    train_loss = train_total_loss / num_train_batches
                 #   tf.profiler.experimental.stop()
                    # Insert train loss and train accuracy after one epoch in corresponding lists
                    train_epoch_loss.append(train_loss)
                    train_epoch_accuracy.append(self.train_accuracy.result())
                    f.write('\t'.join(['Epoch: {}'.format(epoch + 1), 'Train Loss: {}'.format(train_loss),
                                       'Total Train Loss: {}'.format(train_total_loss),
                                       'Train batches: {}'.format(num_train_batches),
                                       'Train accuracy: {}'.format(self.train_accuracy.result()), '\n']))

                    with self.hparams.train_summary_writer.as_default():
                        tf.summary.scalar('loss', train_loss, step=epoch)
                        tf.summary.scalar('accuracy', self.train_accuracy.result(), step=epoch)
                        self.hparams.train_summary_writer.flush()

                    # evaluate model
                    num_test_batches = distributed_test_epoch(self, strategy)
                    print(f'Number of test batches: {num_test_batches}')
                    # insert test loss and test accuracy after one epoch in corresponding lists
                    test_epoch_loss.append(self.test_loss.result())
                    test_epoch_accuracy.append(self.test_accuracy.result())

                    with self.hparams.test_summary_writer.as_default():
                        tf.summary.scalar('loss', self.test_loss.result(), step=epoch)
                        tf.summary.scalar('accuracy', self.test_accuracy.result(), step=epoch)
                        self.hparams.test_summary_writer.flush()

                    f.write('\t'.join(['Epoch: {}'.format(epoch + 1),
                                       'Test Loss: {}'.format(self.test_loss.result()),
                           'Test accuracy: {}'.format(self.test_accuracy.result()), '\n']))

                    if epoch % 3 == 0:
                        self._model.save(
                            self.checkpoint_path + 'epoch-{0}-batch-{1}.ckpt'.format(epoch + 1, num_train_batches), save_format='tf')

                    self.on_epoch_end(epoch, num_train_batches)

                    # Get filtered results, learning curves, ROC and recall precision curves after last epoch
                    if epoch == self.hparams.epochs - 1 or self.stop_training:

                        #                    with open(os.path.join(self.hparams.output, 'metrics.txt'), 'a+') as out:
                        #                        out.write(metrics.classification_report(self.true_classes, self.predicted_classes,
                        #                                                                 target_names=list(self.hparams.class_mapping.values()),
                        #                                                                 digits=3, zero_division=0))
                        #                        out.write('\nConfusion matrix:\n {}\n'.format(
                        #                             metrics.confusion_matrix(self.true_classes, self.predicted_classes)))

                        # create learning curves
                        LearningCurvesPlot(self, train_epoch_loss, train_epoch_accuracy, test_epoch_loss,
                                           test_epoch_accuracy, epoch + 1)

                        self._model.save(self.checkpoint_path + 'saved_model_epoch-{0}-batch-{1}'.format(epoch + 1, num_train_batches), save_format='tf')
                        self._model.save_weights(self.checkpoint_path +
                                                 'epoch-{0}-batch-{1}.ckpt'.format(epoch + 1, num_train_batches))
                        if self.found_min:
                            self._model.set_weights(self.best_weights)
                            self._model.save_weights(self.checkpoint_path + 'minloss.ckpt')
                            # with open(os.path.join(self.hparams.output, 'metrics.txt'), 'a+') as out:
                            #     out.write('\nLowest validation loss epoch: {}\n'.format(self.min_epoch + 1))
                            #     out.write('Test loss: {}\tTest accuracy: {}\n'.format(
                            #         test_epoch_loss[self.min_epoch], test_epoch_accuracy[self.min_epoch]))
                            #     # Print report on precision, recall, f1-score of lowest validation loss
                            #     out.write(metrics.classification_report(self.min_true_class, self.min_pred_class,
                            #                                             target_names=list(
                            #                                                 self.hparams.class_mapping.values()),
                            #                                             digits=3, zero_division=0))
                            #     out.write('\nConfusion matrix for lowest validation loss:\n {}'.format(
                            #         metrics.confusion_matrix(self.min_true_class, self.min_pred_class)))
                        break

                    # Resets all of the metric (train + test accuracies + test loss) state variables.
                    self.train_accuracy.reset_states()
                    self.test_accuracy.reset_states()
                    self.test_loss.reset_states()
                    self.val_loss = test_epoch_loss


            elif self.hparams.mode == 'testing':
                # evaluate model with reads simulated from genomes present in training set
                distributed_test_epoch(self, strategy)
                predictions = np.asarray(self.predictions)
                predicted_classes = np.asarray(self.predicted_classes)
                read_label = np.asarray(self.read_ids)
                print(f'{predicted_classes.shape} - {predictions.shape} - {read_label.shape}')
                num_batches = predicted_classes.shape[0]
                # save predictions to numpy file
                np.save(os.path.join(self.hparams.output, f'{self.hparams.mode}-predictions-numbatches{num_batches}.npy'), predictions)
                # generate output files
                for j in range(num_batches):
                   # print(f'batch: {j}')
                    batch_pred_classes = predicted_classes[j]
                    batch_predictions = predictions[j]
                    batch_read_label = read_label[j]
                    for i in range(len(batch_pred_classes)):
                        label = batch_pred_classes[i]
                        cs = batch_predictions[i][label]
                        genome_label = batch_read_label[i]
                        logits = batch_predictions[i].tolist()
                        f.write(f'{genome_label[0]}\t{label}\t{self.hparams.class_mapping[str(label)]}\t{cs}\t')
                        #for item in logits:
                        #    f.write(f'{item}\t')
                        f.write('\n')
                f.close()

            elif self.hparams.mode == 'inference':
                # start inference
                distributed_test_epoch(self, strategy)
                read_ids = np.asarray(self.read_ids)
                predictions = np.asarray(self.predictions)
                num_batches = predictions.shape[0]
                # generate output files
                for j in range(num_batches):
                    batch_predictions = predictions[j]
                    batch_read_ids = read_ids[j]
                    for i in range(len(batch_predictions)):
                        label, cs = self.adjust_predictions(batch_predictions[i])
                        ri = batch_read_ids[i][0].decode('utf-8')
                        f.write(f'{ri}\t{label}\t{cs}\t{self.hparams.integer_mapping[label]}\n')
                f.close()

    # Custom callback implementation
    def on_epoch_end(self, epoch, num_train_batches):
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
            #            self.min_true_class = self.true_classes
            #            self.min_pred_class = self.predicted_classes
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
