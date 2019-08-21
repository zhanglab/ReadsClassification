
# coding: utf-8

# In[2]:


# Implementation of an artificial neural network
# Load all necessary libraries
from __future__ import division, print_function
import sys
import os
import random
import math
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import keras
from keras import regularizers
from operator import add
from datetime import datetime 
# import the inspect_checkpoint library
from tensorflow.python.tools import inspect_checkpoint as chkp


# In[3]:


# Function that returns a dictionary of all kmers per reads and a list of all unique kmers in dataset
def get_kmers(k_value, dict_seq):
    unique_kmers = set()
    reads_list_kmer = {} # key = read id, value = list of all consecutive kmers in reads (#kmers in read = 150 - k + 1)
    list_unusual_reads = []
    for read, seq in dict_seq.items():  
        reads_list_kmer[read] = []
        for n in range(len(seq)):
            if seq[n] == 'N':
#                 print('Read qui fait chier: {}'.format(read))
                if read not in list_unusual_reads:
                    list_unusual_reads.append(read)
            if n < len(seq) - (k_value - 1):
                kmer = str(seq[n:n+k_value])
                reads_list_kmer[read].append(kmer)
                if kmer not in unique_kmers:
                    unique_kmers.add(kmer)
    # remove all unusual reads from reads_list_kmer
    print(len(list_unusual_reads))
    print(len(reads_list_kmer))
    for x in range(len(list_unusual_reads)):
        del reads_list_kmer[list_unusual_reads[x]]
    print(len(reads_list_kmer))
            
    
    return reads_list_kmer, unique_kmers


# In[4]:


def create_matrix(reads_df, sequence_length, reads_list_kmer, kmers_mapping):
    # Create matrix of zeros, where each row corresponds to a sequence of size of sequence_length
    sequences = np.zeros((reads_df.shape[0], sequence_length), dtype=int)
    # Fill the index of words in each sequence from the right-hand side of the matrix
    for i in range(reads_df.shape[0]):
    #     print(reads_order_df.loc[i,'read_id'])
        # Get list of k-mers for read on row i of dataframe
        list_kmers = reads_list_kmer[reads_df.loc[i,'read_id']]  
        # convert list of kmers into a list of their mapped integers
        list_int = [kmers_mapping.get(key) for key in list_kmers]
    #     print(list_int)
        # transform list into numpy array
        array_int = np.asarray(list_int)
        # Get the first n (sequence_length) kmers and flip array (?)
    #     print(array_int)
        array_int = np.flip(array_int[:sequence_length], axis=0)
        # Replace array in corresponding position in sequences matrix
        sequences[i] = array_int
    return sequences


# In[5]:


def split_dataset(reads_df):
    # For each order get indices of reads in dataframe reads_order_df
    list_orders = reads_df.order.unique().tolist()
    # Enumerate the class labels starting at 0
    class_mapping = {label:idx for idx, label in enumerate(list_orders)}
    print(class_mapping, file=sys.stderr)
    train_idx = []
#     test_idx = []
    for i in range(len(list_orders)):
        # Get list of indices in dataframe of order
        list_indices = reads_df.index[reads_df['order'] == list_orders[i]].tolist()
        # Convert name to integer value given in class_mapping dictionary
        reads_df.loc[list_indices,'order'] = class_mapping[list_orders[i]]
        # shuffle list of indices
#         random.shuffle(list_indices)
        # select 70% of indices for the training set 
#         num_genomes_train = int(0.7*len(list_indices))
        num_genomes_train = int(1*len(list_indices))
        train_idx += list_indices[:num_genomes_train]
        # select the remaining 30% of indices for the test set 
#         test_idx += list_indices[num_genomes_train:]
    
    return train_idx, class_mapping
#     return train_idx, test_idx, class_mapping


# In[6]:


# Function to generate batches of data
def create_batch_generator(x, y=None, batch_size=64):
    # get the number of batches of size 64 given the size of the training dataset
    n_batches = len(x)//batch_size # 2 / will round the float number to an integer
    # create empty array to store y values in case it is not none
    y_copy = np.empty([1, n_batches*batch_size])
    if y is not None:
        y_copy = np.array(y)
        y_copy = y_copy[:n_batches*batch_size]
    # create batch from reduced training set 
    x_copy = np.array(x)
    x_copy = x_copy[:n_batches*batch_size]
    for i in range(0, x_copy.shape[0], batch_size):
        if y is not None:
            yield (x_copy[i:i+batch_size, :], y_copy[i:i+batch_size])
        else:
            yield x_copy[i:i+batch_size, :]


# In[8]:


def get_reads_seq(path):
    reads_seq = {} # keys = read_id, value = read sequence
    reads_sizes = {} # keys = size, value = # of reads with that size
    file_datasets = path + 'list_datasets.txt'
    with open(file_datasets, 'r') as list_datasets:
        for dataset in list_datasets:
            dataset = str(dataset.strip('\n'))
            fList = os.listdir(path + dataset)
            with open(path + dataset +  '/' + 'list_genomes.txt','r') as list_genomes:
                num_read = 0
                for line in list_genomes:
                    line = str(line.strip('\n'))
                    newline = line.split('\t')
                    order_name = str(newline[0])
                    genome_id = str(newline[1])
        #             order_genome[genome_id] = order_name
                    # Open fastq file with reads simulated with genome fasta sequence
                    file_fastq = path + dataset +  '/' + order_name + '_reads.fq'
                    list_reads = []
                    with open(file_fastq, 'r') as reads_f:
                        num_line = 0
                        read_id = str()
                        for line in reads_f: 
                            line = str(line.strip('\n'))
                            if num_line == 1:
                                seq = line
                                reads_seq[read_id] = seq
                                num_read += 1
                                num_line = 0
                                read_id = str()
                            if line[0:4] == '@S0R':
                                read_id = line + '_' + genome_id
                                list_reads.append(line[4])
                                num_line = 1

    return reads_seq


# In[9]:


def get_reads_info(path, dict_reads_updated):
    reads_order_dict = {} # keys = read id and values - name of order
    order_genome = {} # keys = genome_id, value = order
    file_datasets = path + 'list_datasets.txt'
    with open(file_datasets, 'r') as list_datasets:
        num_read = 0
        for dataset in list_datasets:
            dataset = str(dataset.strip('\n'))
            fList = os.listdir(path + dataset)
            with open(path + dataset +  '/' + 'list_genomes.txt','r') as list_genomes:
                for line in list_genomes:
                    line = str(line.strip('\n'))
                    newline = line.split('\t')
                    order_name = str(newline[0])
                    genome_id = str(newline[1])
                    order_genome[genome_id] = order_name
                    # Open fastq file with reads simulated with genome fasta sequence
                    file_fastq = path + dataset +  '/' + order_name + '_reads.fq'
                    list_reads = []
                    with open(file_fastq, 'r') as reads_f:
                        read_id = str()
                        for line in reads_f: 
                            line = str(line.strip('\n'))
                            if line[0:4] == '@S0R':
                                read_id = line + '_' + genome_id
                                if read_id in dict_reads_updated:
                                    reads_order_dict[num_read] = [read_id, order_name]
                                    num_read += 1

    return reads_order_dict


# In[10]:


def model_performance(X_test, Y_test, rnn, class_mapping):
    # Use trained model for predicting the class labels on the test set
    preds = rnn.predict(X_test)
    # Get the ground values of sequences labels (first n=number of labels)
    Y_true = Y_test[:len(preds)]
    print(' Test Accuracy: %.3f' % (np.sum(preds == Y_true) / len(Y_true)), file=sys.stderr)
    # Create confusion matrix for each validation set
    cm = pd.DataFrame(0, columns=list(np.unique(Y_true)), index=list(np.unique(Y_true)))
    # Fill cm
    for i in range(len(Y_true)):
        cm.iloc[cm.index.get_loc(Y_true[i]),cm.columns.get_loc(preds[i])] += 1
    print(cm, file=sys.stderr)
    print(np.unique(Y_true), file=sys.stderr)
    list_labels = np.unique(Y_true)
    # compute recall, precision for different labels
    for j in range(len(list_labels)):
        # get name of labels
        label = str()
        for order, integer in class_mapping.items():
            if integer == list_labels[j]:
                label = order
        true_positives = cm.iloc[list_labels[j],list_labels[j]]
        false_positives = cm.iloc[:,list_labels[j]].sum() - true_positives
        false_negatives = cm.iloc[list_labels[j],:].sum() - true_positives
        recall = true_positives / (false_negatives + true_positives)
        precision = true_positives / (false_positives + true_positives)
        # Print out Recall and Precision for said order
        print('Recall %s: %.2f%%' % (label, 100*recall), file=sys.stderr)
        print('Precision %s: %.2f%%' % (label, 100*precision), file=sys.stderr)


# In[11]:


def get_all_kmers(list_nt, prefix, n, k, list_kmers):
    if k == 0 :
        list_kmers.append(prefix)
        return list_kmers
    
    for i in range(n):
        newPrefix = prefix + list_nt[i]
        get_all_kmers(list_nt, newPrefix, n, k-1, list_kmers)


# In[21]:


def main():
    # Model parameters
    embedding_size = 6
    num_layers = 1
    lstm_size = 256
    learning_rate = 0.001
    num_epochs = 30
    batch_size = 64
    k_value = 10
    num_kmers = 4**k_value
    num_classes = 8
    list_kmers = []
    list_nt = ['A', 'T', 'C', 'G']
    get_all_kmers(list_nt, "", len(list_nt), k_value, list_kmers)
#     print('List of kmers: {}'.format(list_kmers), file=sys.stderr)
    print('Number of {}-mers in list_kmers: {}'.format(k_value ,len(list_kmers)))
    # For each kmer assign an integer randomly between 0 and number of unique kmers in dataset
    # generate a list of integers
    list_num = list(range(0,num_kmers))
    # Assign integer to each unique kmers
    kmers_mapping = dict(zip(list_kmers, list_num))
#     reads_order_dict, reads_seq = get_reads('/glade/u/home/ccres/data/fastq_files_2/Reads_7_orders_1X/')
#     reads_order_dict, reads_seq = get_reads('/Users/Cissou/Desktop/fastq_files_3/Reads_8_orders_1X_1/')
#     reads_order_dict, reads_seq = get_reads('/users/ccres/data/ccres/rnn_meta_classifier/fastq_files_6/Reads_7_orders_7X/')
    # Get sequences for each read
#     reads_seq = get_reads_seq('/Users/Cissou/Desktop/fastq_files_3/Data/')
    reads_seq = get_reads_seq('/users/ccres/data/ccres/rnn_meta_classifier/fastq_files_7/Data/')
    print('Number of reads: {}'.format(len(reads_seq)))
    print('k value: {0}'.format(k_value))
#     # Get kmers from removes all reads with unusual characters
#     # Get kmers
    reads_list_kmer, unique_kmers = get_kmers(k_value, reads_seq)
# #     print(unique_kmers)
#     # Print out total number of kmers in dataset 
#     n_kmers = len(unique_kmers)
#     print('Number of {0}-mers in dataset: {1}'.format(k,n_kmers), file=sys.stderr)
#     print('Number of {0}-mers: {1}'.format(k,num_kmers), file=sys.stderr)
#     # get info on reads dataset updated
#     reads_order_dict = get_reads_info('/Users/Cissou/Desktop/fastq_files_3/Reads_8_orders_1X_1/',reads_list_kmer)
    reads_order_dict = get_reads_info('/users/ccres/data/ccres/rnn_meta_classifier/fastq_files_7/Data/',reads_list_kmer)
    print('Number of reads: {}'.format(len(reads_order_dict)))
#     # Create dataframe of reads
#     # Create list of names for columns in dataframe
    headers = ['read_id'] + ['order']
    # Create dataframe
    reads_order_df = pd.DataFrame.from_dict(reads_order_dict, orient='index')
    reads_order_df.columns = headers
    list_orders = reads_order_df.order.unique().tolist()
    class_mapping = {label:idx for idx, label in enumerate(list_orders)}
    print(class_mapping)
    print(reads_order_df.head())
    print(reads_order_df.tail())
    print(reads_order_df.order.unique())
    print(reads_order_df.shape)
    
    print("\nProcess reads: START -- {}".format(datetime.now()))
    # Set size of vector representing each read
    sequence_length = int(150 - k_value + 1)
    print('Vector size with k = {0}: {1}'.format(k_value, sequence_length))
    # Create matrix of vectors
    sequences = create_matrix(reads_order_df, sequence_length, reads_list_kmer, kmers_mapping)
    print("\nProcess reads: END -- {}".format(datetime.now()))
    # Convert order names to integers in dataframe
    print('List of orders in training set: {}'.format(reads_order_df.order.unique()))
    for n in range(len(list_orders)):
        # Get list of indices in dataframe of order
        print(list_orders[n])
        list_indices = reads_order_df.index[reads_order_df['order'] == list_orders[n]].tolist()
        print(len(list_indices))
        # Convert name to integer value given in class_mapping dictionary
        reads_order_df.loc[list_indices,'order'] = class_mapping[list_orders[n]]
    print('List of orders in training set: {}'.format(reads_order_df.order.unique()))
#     # Get entire dataset into training set
    train_idx = reads_order_df.index.values.tolist()
    print('Size of training set: {}'.format(len(train_idx)))
    random.shuffle(train_idx)
    X_train = sequences[train_idx,:]
    Y_train = reads_order_df.loc[train_idx,'order'].values
    print('Size of vector X train: {}'.format(len(X_train)))
    print('Size of vector Y train: {}'.format(len(Y_train)))
    
    # reset graph
    tf.reset_default_graph()
    
    # Create new empty computation graph
    g = tf.Graph()
    # Define model
    with g.as_default():
        # Set a graph-level seed to make the random sequences generated by all operations be repeatable across sessions
        tf.set_random_seed(123)
        with tf.device('/device:GPU:0'):
            # Define the placeholders for holding the input values: sequences of unique kmers (tf_x) 
            # and the response values (tf_y)
            tf_x = tf.placeholder(dtype=tf.int32, shape=[batch_size, sequence_length], name='tf_x')
            print(tf_x)
            tf_y = tf.placeholder(dtype=tf.int32, shape=[batch_size], name='tf_y')
            print(tf_y)
            tf_keepprob = tf.placeholder(tf.float32, name='tf_keepprob')
            print(tf_keepprob)
            # Convert labels to a one-hot representation (necessary to use the cost function)
            y_onehot = tf.one_hot(indices=tf_y, depth=num_classes)
            print(y_onehot, file=sys.stderr)
            # Create embedding layer: Create a matrix of size [n_kmers x embedding_size] as a tensor variable and initialize its elements
            # randomly with floats between [-1,1]
            print("\nCreate embedding layer: START")
            print(datetime.now())
            embedding = tf.Variable(tf.random_uniform((num_kmers, embedding_size), minval=-1, maxval=1), name='embedding')
            print(embedding)
            embed_x = tf.nn.embedding_lookup(embedding, tf_x, name='embeded_x')
            print(embed_x)
            print("\nCreate embedding layer + embed_x: DONE")
            print(datetime.now())
            # define LSTM cell and stack them together
            # BasicLSTMCell wrapper class that defines LSTM cells which can be stacked together to form a multilayer RNN
            # using the MultiRNNCell wrapper class, apply dropout (NN regularization) to each layer
            # make a list using python list comprehension of LSTM cells according to the desired number of RNN layers
            cells = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.DropoutWrapper(
                    tf.nn.rnn_cell.LSTMCell(lstm_size), 
                    output_keep_prob=tf_keepprob)
                    for i in range(num_layers)])
    #         tf.nn.rnn_cell.LSTMCell(name='basic_lstm_cell')
            # Define the initial state (there are 3 types of inputs in LSTM cells: input data x (embed_x data tensor), 
            # activations of hidden units
            # from the previous time step h and the cell state from the previous time step C)
            # when we start processing a new input sequence, we initialize the cell states to zero state
            initial_state = cells.zero_state(
                batch_size, tf.float32)
            print(' << initial state >>', initial_state)
        with tf.device('/device:GPU:1'):
            # the tf.nn.dynamic_rnn function pulls the embedded data, the RNN cells and their initial states and creates
            # a pipeline for them according to the architecture of LSTM cells
            # We store the final state to use  as the initial state of the next mini-batch of data
    #         lstm_outputs, self.final_state = tf.nn.dynamic_rnn(
    #             cells, embed_x, initial_state=self.inital_state)
            lstm_outputs, final_state = tf.nn.dynamic_rnn(
                cells, embed_x, initial_state=initial_state)
            # the tf.nn.dynamic_rnn function returns a tuple containing the activations of the RNN cells: outputs and their
            # final state: state. The output is a 3 dimensional tensor with the following shape:
            # lstm_outputs shape: [batch_size, max_time, cells.output_size]
            print('\n << lstm_output >>', lstm_outputs)
            print('\n << final state >>', final_state)
            print(lstm_outputs[:, -1])
            # Pass outputs to a connected layer to get logits 
            logits = tf.layers.dense(inputs=lstm_outputs[:, -1], units=num_classes, activation=None, name='logits')
            print('\n << logits >>', logits)
            y_proba = tf.nn.softmax(logits, name='probabilities')
            predictions = {
                'labels': tf.argmax(logits, axis=1, name='labels'),
                'probabilities': y_proba
            }

            print('\n << predictions >>', predictions)
            # Define the cost function
            cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=y_onehot),
                name='cost')
            print('\n << cost >>', cost)
            # Define the optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(cost, name='train_op')

        # Create saver object
        saver = tf.train.Saver()
        # Returns an Operation that initializes global variables in the graph
        init_op = tf.global_variables_initializer()
    
    
    # Train model
    print("\nStart training: {}".format(datetime.now()))
    with tf.Session(graph=g) as sess:
            sess.run(init_op)
            iteration = 1
            plot_cost = []
            preds_all = []  # keep test accuracy after each epoch for testing set
            # Train model 10 times with different sets of data 
            for epoch in range(num_epochs):
                # start from the zero states of RNN cells as our current state
                state = sess.run(initial_state)
                training_costs = []
                batch_generator = create_batch_generator(X_train, Y_train, batch_size=64)
                for batch_x, batch_y in batch_generator:
                    # feeding the current state along with the data batch_x and their labels batch_y
                    feed = {'tf_x:0': batch_x,
                                'tf_y:0': batch_y,
                                'tf_keepprob:0': 0.5,
                                initial_state: state}
                    # At the end of a mini-batch, update the state to be the final state
                    batch_loss, _, state = sess.run(['cost:0', 'train_op', final_state], feed_dict=feed)
                    training_costs.append(batch_loss)
                    # Print train loss of model every 20 epochs
#                     if iteration % 20 == 0:
                    print("Epoch: %d/%d Iteration: %d "
                                 "| Train loss: %.5f" %(epoch + 1, num_epochs, iteration, batch_loss))
                    print(' -- Epoch %2d ''Avg. Training Loss: %.4f' % (epoch+1, np.mean(training_costs)))
                    iteration += 1
                # Get average training costs/test accuracy in batches for each epoch
                plot_cost.append(np.mean(training_costs))  
                # Save model after every epoch
#                 if (epoch+1)%3 == 0:
#                   self.saver.save(sess, "model/SpeciesRNN-%d.ckpt" % epoch)
                saver.save(sess, "model-8orders-sw-k{0}-cov1-100%-WO-Class/SpeciesRNN-sw-8orders-{0}-{1}.ckpt".format(k_value,epoch))
                # Test model with testing set
                preds_batches = []
                for ii, batch_x_1 in enumerate(create_batch_generator(X_train, None, batch_size=batch_size), 1):
                    feed = {'tf_x:0':batch_x_1, 'tf_keepprob:0': 1.0, initial_state: state}
                    pred, prob, test_state = sess.run(['labels:0', 'probabilities:0', state], feed_dict=feed)
                    preds_batches.append(pred)
                preds_all.append(np.mean(preds_batches))
                
            # Generate plot to visualize the training cost after each epoch
            plt.clf()
            plt.plot(range(1,len(plot_cost) + 1), plot_cost)
            plt.ylabel('Training cost',fontsize=15)
            plt.xlabel('Epoch',fontsize=15)
            plt.savefig('/users/ccres/data/ccres/run/Training_Loss_{0}_k{1}_8orders_sw_RNN_cov1_100%-WO-Class.png'.format(learning_rate, k_value))
            # Generate plot to visualize the training cost after each epoch
            plt.clf()
            plt.plot(range(1,len(preds_all) + 1), preds_all)
            plt.ylabel('Test accuracy',fontsize=15)
            plt.xlabel('Epoch',fontsize=15)
            plt.savefig('/users/ccres/data/ccres/run/Test_Accuracy_TrainingSet_{0}_k{1}_8orders_sw_RNN_cov1_100%-WO-Class.png'.format(learning_rate, k_value))
    print("\nEnd training: {}".format(datetime.now()))


# In[22]:


if __name__ == "__main__":
    main()

