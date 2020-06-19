#!/usr/bin/env python
# coding: utf-8

# Import all necessary libraries
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

# generate random numbers with numpy
np.random.seed(42)
import tensorflow as tf
# Check the devices used by tensorflow
from tensorflow.python.client import device_lib
tf.random.set_seed(1234)
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import datetime
import pandas as pd
import keras
from sklearn import metrics
#import optuna

print(tf.__version__, file=sys.stderr)
print(device_lib.list_local_devices(), file=sys.stderr)


class MultiGPUs(object):
    def __init__(self, epochs, model, batch_size, strategy, checkpoint_path, hiddensize, class_mapping, embedding_size):
        self.epochs = epochs
        self.global_batch_size = batch_size
        self.strategy = strategy
        self.loss_object = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.optimizer = tf.keras.optimizers.Adamax(learning_rate=0.01)
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
        self.test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.hidden_size = hiddensize
        self.class_mapping = class_mapping
        self.embedding_size = embedding_size
        # Create lists to store true classes and predicted classes
        self.true_classes = list()
        self.predicted_classes = list()
        self.probability_threshold = 0.5
        self.probabilities = list()
        self.unclassified = int()

    '''def get_hparams(trial):
        kwargs = {}
        kwargs['embedding_size'] = trial.suggest_int(20, 100)
        kwargs['hidden_size'] = trial.suggest_int(20, 100)
        kwargs['global_batch_size'] = trial.suggest_int(32, 100)
        kwargs['dropout_rate'] = trial.suggest_float(0.5, 0.8)
        return kwargs'''

    def decay(self, epoch):
        if epoch >= 100:
            return 0.00001
        elif epoch >= 50:
            return 0.0001
        return 0.01

    def compute_loss(self, label, predictions):
        loss = tf.reduce_sum(self.loss_object(label, predictions)) * (1.0 / self.global_batch_size)
        return loss

    # Define one train step
    def train_step(self, inputs):
        reads, labels = inputs
        with tf.GradientTape() as tape:
            # Get the predictions as probabilities
            predictions = self.model(reads, training=True)
            #            print('predictions: {}'.format(predictions))
            # Calculate the loss
            loss = self.compute_loss(labels, predictions)
        # compute the gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # update the weights
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        # update the accuracy
        self.train_accuracy.update_state(labels, predictions)

        return loss

    # Define one test step
    def test_step(self, inputs):
        reads, labels = inputs
        predictions = self.model(reads, training=False)
        # Get true and predicted labels
        pred_classes = tf.keras.backend.argmax(predictions)
        t_classes = tf.keras.backend.argmax(labels)
        # Get highest probability in vector
        prob = tf.reduce_max(predictions, axis=1)
        #        print('probabilities: {}'.format(prob))

        # convert to proto tensor
        proto_tensor_t_classes = tf.make_tensor_proto(t_classes)
        proto_tensor_pred_classes = tf.make_tensor_proto(pred_classes)
        proto_tensor_prob = tf.make_tensor_proto(prob)
        self.true_classes += tf.make_ndarray(proto_tensor_t_classes).tolist()
        self.predicted_classes += tf.make_ndarray(proto_tensor_pred_classes).tolist()
        self.probabilities += tf.make_ndarray(proto_tensor_prob).tolist()

        test_loss = self.loss_object(labels, predictions)
        self.test_accuracy.update_state(labels, predictions)
        self.test_loss.update_state(test_loss)
        # print('accuracy: {}'.format(self.test_accuracy.result()))

    def LearningCurvesPlot(self, train_epoch_loss, train_epoch_accuracy, test_epoch_loss, test_epoch_accuracy):
        plt.clf()
        fig = plt.figure(dpi=500)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        # Add some extra space for the second axis at the bottom
        fig.subplots_adjust(bottom=0.2)
        fig.set_size_inches(7, 7)
        list_num_epochs = list(range(1, self.epochs + 1, 1))
        print(list_num_epochs, file=sys.stderr)
        # x_coords = [i*self.strategy.num_replicas_in_sync*num_train_batches for i in range(1,self.epochs+1)]
        x_coords = [i for i in range(1, self.epochs + 1)]
        print(x_coords, file=sys.stderr)
        ax2.plot(list_num_epochs, test_epoch_accuracy, color='black', linewidth=2.0,
                 label='Average Validation Accuracy')
        ax2.plot(list_num_epochs, train_epoch_accuracy, color='green', linewidth=2.0, label='Average Training Accuracy')
        ax1.plot(list_num_epochs, test_epoch_loss, color='red', linewidth=2.0, label='Validation Loss')
        ax1.plot(list_num_epochs, train_epoch_loss, color='blue', linewidth=2.0, label='Training Loss')
        ax1.set_ylabel('Loss (%)', fontsize=14, labelpad=12)
        ax1.set_xlabel('Number of epochs', fontsize=14, labelpad=12)
        ax2.set_ylabel('Average Accuracy (%)', fontsize=14, labelpad=12)
        ax2.ticklabel_format(useOffset=True, style='sci')
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                   fancybox=True, shadow=False, ncol=5)
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
                   fancybox=True, shadow=False, ncol=5)
        # Add vertical lines to represent each epoch
        for xcoord in x_coords:
            plt.axvline(x=xcoord, color='gray', linestyle='--')
        plt.savefig(os.getcwd() + '/run/LearningCurves-{0}ul-{1}emb.png'.format(self.hidden_size, self.embedding_size),
                                                                                                bbox_inches='tight')

    def GetMetrics(self, cm, rank, listtaxa):
        print('Rank: {}'.format(rank))
        template = ('Taxon: {}, Recall: {}, Precision: {}')
        tp = float()
        fp = float()
        fn = float()
        tn = float()
        recall = float()
        precision = float()
        accuracy = float()
        print(cm)
        # Get the FP, TP, FN and TN for each class
        for j in range(cm.shape[0]):
            true_pos = cm.iloc[j, j]
            tp += cm.iloc[j, j]
            false_pos = cm.iloc[:, j].sum() - true_pos
            fp += false_pos
            false_neg = cm.iloc[j, :].sum() - true_pos
            fn += false_neg
            true_neg = cm.values.sum() - false_pos - false_neg - true_pos
            tn += true_neg
            recall += true_pos / (false_neg + true_pos)
            precision += true_pos / (true_pos + false_pos)
            print(template.format(listtaxa[j], round(float(true_pos) / (true_pos + false_neg), 3),
                                  round(float(true_pos) / (true_pos + false_pos), 3)))

        recall_micro = tp / (tp + fn)
        precision_micro = tp / (tp + fp)
        recall_macro = recall / cm.shape[0]
        precision_macro = precision / cm.shape[0]
        print(
            'Test Accuracy entire test set {0} level: {1}'.format(rank, round(float(tp + tn) / (tp + tn + fn + fp), 3)))
        print('Macro precision {0} level: {1}'.format(rank, precision_macro))
        print('Micro precision {0} level: {1}'.format(rank, precision_micro))
        print('Macro recall {0} level: {1}'.format(rank, recall_macro))
        print('Micro recall {0} level: {1}'.format(rank, recall_micro))

    def GetFilteredResults(self):
        listspecies = list(self.class_mapping.values())
        #        listgenus = ['Polaribacter', 'Reinekea', 'Colwellia', 'Paraglaciecola', 'Octadecabacter', 'Sulfitobacter', 'Halocynthiibacter', 'Pseudoalteromonas', 'Vibrio']
        listgenus = ['Polaribacter', 'Reinekea', 'Colwellia']
        filteredpred = list()
        filteredtrue = list()
        cm = pd.DataFrame(0, columns=listspecies, index=listspecies)
        cm_filtered = pd.DataFrame(0, columns=listspecies, index=listspecies)
        cm_genus = pd.DataFrame(0, columns=listgenus, index=listgenus)
        for i in range(len(self.true_classes)):
            true_class = self.true_classes[i]
            pred_class = self.predicted_classes[i]
            # Get species name
            predspecies = self.class_mapping[pred_class]
            truespecies = self.class_mapping[true_class]
            # Get genus name
            predgenus = self.dictgenus[predspecies]
            truegenus = self.dictgenus[truespecies]
            # Get probability
            prob = self.probabilities[i]
            # add entry into confusion matrix if probability above threshold
            if prob > self.probability_threshold:
                filteredpred.append(pred_class)
                filteredtrue.append(true_class)
                cm_filtered.iloc[cm_filtered.index.get_loc(truespecies), cm_filtered.columns.get_loc(predspecies)] += 1
                cm_genus.iloc[cm_genus.index.get_loc(truegenus), cm_genus.columns.get_loc(predgenus)] += 1
            else:
                self.unclassified += 1
            cm.iloc[cm.index.get_loc(truespecies), cm.columns.get_loc(predspecies)] += 1
        print('Test Accuracy entire test set species level - filtered predictions: %.2f%%' % (
                    100 * np.sum(filteredpred == filteredtrue) / len(filteredpred)))
        print('Number of Unclassified reads: {}'.format(self.unclassified))
        print('Total number of reads tested 1: {}'.format(len(self.true_classes)))
        print('Total number of reads tested 2: {}'.format(len(self.predicted_classes)))
        print('Total number of reads tested 3: {}'.format(len(self.probabilities)))
        print('Results from unfiltered predictions at the species level:')
        self.GetMetrics(cm, 'Species', listspecies)
        print('Results from filtered predictions at the species level:')
        self.GetMetrics(cm_filtered, 'Species', listspecies)
        print('Results from filtered predictions at the genus level:')
        self.GetMetrics(cm_genus, 'Genus', listgenus)

    def custom_training_loop(self, train_dist_set, test_dist_set, strategy):
        train_epoch_loss = []
        train_epoch_accuracy = []
        test_epoch_loss = []
        test_epoch_accuracy = []

        def distributed_train_epoch(ds):
            total_loss = 0.0
            num_train_batches = 0.0
            for one_batch in ds:
                # Get the loss from each GPU/device
                per_replica_loss = strategy.experimental_run_v2(
                    self.train_step, args=(one_batch,))
                total_loss_batch = strategy.reduce(
                    tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None).numpy()
                total_loss += total_loss_batch
                num_train_batches += 1

                if num_train_batches % 100 == 0:
                    print('Train batch #: {0} - Train Accuracy: {1} - Train Loss: {2}'.format(num_train_batches,
                                                                                              self.train_accuracy.result(),
                                                                                              total_loss_batch), file=sys.stderr)

            return total_loss, num_train_batches

        def distributed_test_epoch(ds, num_elements_test_dist_dataset):
            num_test_batches = 0.0
            for one_batch in ds:
                num_test_batches += 1
                if num_test_batches < num_elements_test_dist_dataset - 1:
                    strategy.experimental_run_v2(self.test_step, args=(one_batch,))

                if num_test_batches % 100 == 0:
                    print('Test batch #: {}'.format(num_test_batches))
                    print('accuracy: {}'.format(self.test_accuracy.result()))
                    print('test loss: {}'.format(self.test_loss.result()))

        for epoch in range(self.epochs):
            num_elements_train_dist_dataset = 0
            for inputs in train_dist_set:
                num_elements_train_dist_dataset += 1
            num_elements_test_dist_dataset = 0
            for inputs in test_dist_set:
                num_elements_test_dist_dataset += 1
            print('Number of elements in distributed train dataset: {}'.format(num_elements_train_dist_dataset),
                  file=sys.stderr)
            print('Number of elements in distributed test dataset: {}'.format(num_elements_test_dist_dataset),
                  file=sys.stderr)
            self.optimizer.learning_rate = self.decay(epoch)
            print('Learning rate at epoch {0}: {1}'.format(epoch, self.optimizer.learning_rate))
            train_total_loss, num_train_batches = distributed_train_epoch(train_dist_set)

            # insert train loss and train accuracy after one epoch in corresponding lists
            train_epoch_loss.append(train_total_loss / num_train_batches)
            train_epoch_accuracy.append(self.train_accuracy.result())

            # use validation set to get performance
            distributed_test_epoch(test_dist_set, num_elements_test_dist_dataset)
            # insert test loss and test accuracy after one epoch in corresponding lists
            test_epoch_accuracy.append(self.test_accuracy.result())
            test_epoch_loss.append(self.test_loss.result())

            template1 = ('Epoch: {}, Train Loss: {}, Train Accuracy: {}, Test Loss: {}, Test Accuracy: {}')
            template2 = (
                'Epoch: {}, Total Train Loss: {}, Train batches: {}, Train Accuracy: {}, Test Loss: {}, Test Accuracy: {}')
            print(template1.format(epoch + 1, train_total_loss / num_train_batches,
                                   self.train_accuracy.result() * 100,
                                   self.test_loss.result(),
                                   self.test_accuracy.result() * 100))
            print(template2.format(epoch + 1, train_total_loss, num_train_batches,
                                   self.train_accuracy.result() * 100,
                                   self.test_loss.result(),
                                   self.test_accuracy.result() * 100))

            # Get filtered results, learning curves, ROC and recall precision curves after last epoch
            if epoch == self.epochs - 1:
                # Print report on precision, recall, f1-score
                print('Metrics report from sklearn:')
                print(metrics.classification_report(self.true_classes, self.predicted_classes, digits=3, zero_division=0))
                # Plot precision-recall curves
                dict_metrics = metrics.classification_report(self.true_classes, self.predicted_classes, zero_division=0,
                                                             output_dict=True)
                # for label, metricsvalues in dict_metrics.items():
                #    print('Label: {0} - Metrics: {1}'.format(label, metricsvalues))
                #                self.GetFilteredResults()
                self.LearningCurvesPlot(train_epoch_loss, train_epoch_accuracy, test_epoch_loss, test_epoch_accuracy)
                # save model at the end of last epoch
                self.model.save_weights(
                    self.checkpoint_path + 'V2epoch-{0}-batch-{1}.ckpt'.format(epoch + 10, num_train_batches))
                print('checkpoint path after saving model: {}'.format(self.checkpoint_path), file=sys.stderr)
            # Resets all of the metric (train + test accuracies + test loss) state variables.
            self.train_accuracy.reset_states()
            self.test_accuracy.reset_states()
            self.test_loss.reset_states()
            self.true_classes = list()
            self.predicted_classes = list()
            self.probabilities = list()
            self.unclassified = int()

def CreateModel(hidden_size, num_classes, num_bases, embedding_size, sequence_length):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=num_bases, output_dim=embedding_size, input_length=sequence_length, trainable=True))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.RNN(tf.keras.layers.LSTMCell(hidden_size)))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    print(model.summary())
    return model

def LoadData(num_classes):
    # Load data
    traindata = np.load(os.getcwd() + '/data/train_data_50000_{0}classes.npy'.format(num_classes), allow_pickle=True)
    valdata = np.load(os.getcwd() + '/data/val_data_50000_{0}classes.npy'.format(num_classes), allow_pickle=True)
    #    np.random.shuffle(data)
    #    print(data[0])
    X_train = np.array([i[0] for i in traindata])
    y_train = np.array([i[1] for i in traindata])
    X_test = np.array([i[0] for i in valdata])
    y_test = np.array([i[1] for i in valdata])
    print('Number of reads in training set: {}'.format(X_train.shape), file=sys.stderr)
    print('Number of reads in testing set: {}'.format(X_test.shape), file=sys.stderr)
    return X_train, y_train, X_test, y_test

def GetSetInfo():
    with open('class_mapping.json') as f
        class_mapping = json.load(f)
    listspecies = list(class_mapping.values())
    print('Dictionary mapping Classes to integers: {}'.format(class_mapping))
    return class_mapping, listspecies

def main():
    # get list of genomes in file directory to reads
    class_mapping, listspecies = GetSetInfo()
    num_classes = len(class_mapping)
    num_epochs = 10
    print('number epochs: {0}, number of species: {1}'.format(num_epochs, num_classes))

    X_train, y_train, X_test, y_test = LoadData(num_classes)
    # convert label integers into one hot encoddings
    Y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
    Y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

    # create a strategy to distribute the variables and the graph across multiple GPUs
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync), file=sys.stderr)

    # Set batch size
    batch_size_per_device = (32 * strategy.num_replicas_in_sync)
    global_batch_size = (batch_size_per_device * strategy.num_replicas_in_sync)
    BUFFER_SIZE = len(X_train)

    #study = optuna.create_study(storage='sqlite:///LSTM.db', load_if_exists=True)

    with strategy.scope():
        # Build model
        model = CreateModel(hidden_size, num_classes, 4, 60, 150)
        # Define checkpoint path
        checkpoint_path = os.getcwd() + "/run/LSTM-{0}lu-1LSTM-{1}emb-32bpr-".format(hidden_size, embedding_size)
        # uncomment next 2 lines if training model for another round
        # checkpoint_path_to_load = "/glade/u/home/ccres/run/RNN5/10classes/LSTM-k5-30sl-32lu-1LSTM-5emb-32bpr-V2epoch-9-batch-2553.0.ckpt"
        # model.load_weights(checkpoint_path_to_load)
        # Create trainer object
        trainer = MultiGPUs(num_epochs, model, global_batch_size, strategy, checkpoint_path, hidden_size, class_mapping)
        #study.optimize(trainer, n_trials=100)
        # Slice and batch train and test datasets
        Train_dataset_iter = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(BUFFER_SIZE).batch(
            global_batch_size)
        Test_dataset_iter = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(global_batch_size)
        # Distribute train and test datasets
        train_dist_dataset = strategy.experimental_distribute_dataset(Train_dataset_iter)
        test_dist_dataset = strategy.experimental_distribute_dataset(Test_dataset_iter)
        # Train model
        print("\nStart training: {}".format(datetime.datetime.now()), file=sys.stderr)
        trainer.custom_training_loop(train_dist_dataset, test_dist_dataset, strategy)
        print("\nEnd training: {}".format(datetime.datetime.now()), file=sys.stderr)


if __name__ == "__main__":
    main()