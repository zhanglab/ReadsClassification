#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import all necessary libraries
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
# generate random numbers with numpy 
np.random.seed(42)
#np.set_printoptions(threshold=sys.maxsize)
import tensorflow as tf
# Check the devices used by tensorflow
from tensorflow.python.client import device_lib
tf.random.set_seed(1234)
# import the inspect_checkpoint library
from tensorflow.python.tools import inspect_checkpoint as chkp
import os
import sys
import random
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from operator import add
import datetime
from Bio import SeqIO
import gzip
import re
from contextlib import redirect_stdout



# start core Python generated random numbers in a well-defined state
#random.seed(12345)

# force tensorflow to use single thread
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

print(tf.__version__)
print(device_lib.list_local_devices())
#tf.debugging.set_log_device_placement(True)
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True


class MultiGPUs(object):
    def __init__(self, epochs, model, batch_size, strategy):
        self.epochs = epochs
        self.batch_size = batch_size
        self.strategy = strategy
        self.loss_object = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
        self.test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
        self.test_loss = tf.keras.metrics.Sum(name='test_loss')
        self.model = model
        
#     def decay(self, epoch):
#         if epoch < 3:
#             return 0.1
#         elif epoch >= 3 and epoch < 7:
#             return 0.0001
#         else:
#             return 0.00001
        
    def compute_loss(self, label, predictions):
        # get average loss for one batch: compute the loss for each example in the batch, 
        # sum the losses and then divide by the number of examples in the batch
        loss = tf.reduce_sum(self.loss_object(label, predictions)) * (1.0 / (self.batch_size * self.strategy.num_replicas_in_sync))
#         print('loss1: {}'.format(loss1))
#         print('Loss2: {}'.format(self.loss_object(label, predictions)))
#         print('Loss3: {}'.format(tf.reduce_sum(self.loss_object(label, predictions))))
#        loss2 = tf.nn.compute_average_loss(self.loss_object(label, predictions),global_batch_size=(self.batch_size * self.strategy.num_replicas_in_sync))
#         print('loss4: {}'.format(loss2))
        # add the loss to the global loss and divide by the number of devices available --> get the loss per devices
#         loss += (sum(self.model.losses)) * (1.0 / self.strategy.num_replicas_in_sync)
        return loss
        
    # Define one train step
    def train_step(self, inputs):
        reads, order = inputs
        
        with tf.GradientTape() as tape:
            predictions = self.model(reads, training=True)
            loss = self.compute_loss(order, predictions)
            
#             print('loss5: {}'.format(loss))
            
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        self.train_accuracy.update_state(order, predictions)
#        print('Train accuracy: {}'.format(self.train_accuracy.result()))
        return loss
        
    # Define one test step
    def test_step(self, inputs):
        reads, order = inputs
        
        predictions = self.model(reads, training=False)
        test_loss = self.loss_object(order, predictions)
        print('Test loss per replica: '.format(test_loss))
        self.test_accuracy.update_state(order, predictions)
        self.test_loss.update_state(test_loss)
        
#        print('Test accuracy: {}'.format(self.test_accuracy.result()))
        print('Test loss: {}'.format(self.test_loss.result()))
    
    def LearningCurvesPlot(self,Train_Loss_Epoch, Train_Accuracy_Epoch, Test_Loss_Epoch, Test_Accuracy_Epoch):
        plt.clf()
        fig = plt.figure(dpi=600)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        # Add some extra space for the second axis at the bottom
        fig.subplots_adjust(bottom=0.2)
        fig.set_size_inches(5,5)
        ax2.plot(Test_Accuracy_Epoch, color='black', linewidth=2.0, label='Average Accuracy')
        ax1.plot(Test_Loss_Epoch, color='red', linewidth=2.0, label='Validation Loss')
        ax1.plot(Train_Loss_Epoch, color='blue', linewidth=2.0, label='Training Loss')
        ax1.set_ylabel('Loss (%)', fontsize=14, labelpad=12)
        ax1.set_xlabel('Number of epochs', fontsize=14, labelpad=12)
        ax2.set_ylabel('Average Accuracy (%)', fontsize=14, labelpad=12)
        ax2.ticklabel_format(useOffset=True,style='sci')
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
              fancybox=True, shadow=False, ncol=5)
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
              fancybox=True, shadow=False, ncol=5)
        plt.savefig('/glade/u/home/ccres/run/results.png',bbox_inches='tight')
        plt.show()
        
    def custom_training_loop(self, train_dist_set, test_dist_set, strategy):
        
        def distributed_train_epoch(ds):
            total_loss = 0.0
            num_train_batches = 0.0
            for one_batch in ds:
#                print('Size of one batch: {}'.format(len(one_batch)))
                # Get the loss from each GPU/device
                # tf.distribute.Strategy.experimental_run_v2 returns results from each 
                # local replica in the strategy
                per_replica_loss = strategy.experimental_run_v2(
                            self.train_step, args=(one_batch,))
#                 print('Loss6: {}'.format(per_replica_loss))
                total_loss += strategy.reduce(
                    tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
#                 print('Loss7: {}'.format(total_loss))
#                 print('Number of train batches: {}'.format(num_train_batches))
#                 print('Loss per devices: {}'.format(per_replica_loss))
                num_train_batches += 1
            return total_loss, num_train_batches
        
        def distributed_test_epoch(ds):
            num_test_batches = 0.0
            for one_batch in ds:
                strategy.experimental_run_v2(self.test_step, args=(one_batch,))
                print('total loss test: {}'.format(strategy.experimental_run_v2(self.test_step, args=(one_batch,)))
                print('Loss updated: {}'.format(self.test_loss.result()))
                num_test_batches += 1
#             print('Number of test batches: {}'.format(num_test_batches))
#             print('Test accuracy: {}'.format(self.test_accuracy.result()))
#             print('Test loss: {}'.format(self.test_loss.result()))
            return num_test_batches
        
        Train_Loss_Epoch = []
        Train_Accuracy_Epoch = []
        Test_Loss_Epoch = []
        Test_Accuracy_Epoch = []
        for epoch in range(self.epochs):
#             self.optimizer.learning_rate = self.decay(epoch)
            train_total_loss, num_train_batches = distributed_train_epoch(train_dist_set)
            num_test_batches = distributed_test_epoch(test_dist_set)
            Train_Loss_Epoch.append(train_total_loss/num_train_batches)
            Train_Accuracy_Epoch.append(self.train_accuracy.result())
            Test_Loss_Epoch.append(self.test_loss.result()/num_test_batches)
            Test_Accuracy_Epoch.append(self.test_accuracy.result())
            
            template = ('Epoch: {}, Train Loss: {}, Train Accuracy: {}, Test Loss: {}, Test Accuracy: {}')
            
            print(template.format(epoch+1, train_total_loss/num_train_batches,
                                 self.train_accuracy.result()*100,
                                 self.test_loss.result()/num_test_batches,
                                 self.test_accuracy.result()*100))
            
            # Save the model
            # Create a checkpoint directory to store the checkpoints.
#            checkpoint_dir = '/glade/u/home/ccres/run/training_checkpoints'
#            checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
            
            # Reset the accuracy variable between each epoch
            # if epoch != self.epochs - 1:
            self.train_accuracy.reset_states()
            self.test_accuracy.reset_states()
            self.test_loss.reset_states()
        print(Train_Loss_Epoch)
        print(Test_Loss_Epoch)
        print(Train_Accuracy_Epoch)
        print(Test_Accuracy_Epoch)
        self.LearningCurvesPlot(Train_Loss_Epoch, Train_Accuracy_Epoch,Test_Loss_Epoch, Test_Accuracy_Epoch)



# In[2]:


# Function to shuffle the classes and the sequences and create test and train sets
def CreateSets(ArrayOfVectors, ArrayOfClasses, ArrayOfIDs, sequence_length, num_classes):
    # Convert classes integer to onehot vectors
    ArrayOfOnehot = tf.keras.utils.to_categorical(ArrayOfClasses, num_classes=num_classes, dtype='float32')
    print('ArrayOfOnehot: {}'.format(len(ArrayOfOnehot)))
    # Create a set of indices
    ListOfIndices = list(range(0,len(ArrayOfVectors)))
    # Shuffle list of indices
    random.shuffle(ListOfIndices)
    print('First index: {}'.format(ListOfIndices[0]))
    # Create matrix of zeros, where each row corresponds to a read (vector of kmers)
    NewArrayOfVectors = np.zeros((len(ArrayOfVectors), sequence_length), object)
    NewArrayOfClasses = np.zeros((len(ArrayOfOnehot), num_classes), object)
#    NewArrayOfIDs = np.zeros((len(ArrayOfOnehot), num_classes), object)
#     ListOnehot = ArrayOfOnehot.tolist()
#     ListVectors = ArrayOfVectors.tolist()
#     print(len(ListOnehot))
#     print(len(ListVectors))
#     NewListOnehot = [None] * len(ListOnehot)
#     NewListVectors = [None] * len(ListVectors)
    for i in range(len(NewArrayOfVectors)):
        NewArrayOfVectors[ListOfIndices[i]] = ArrayOfVectors[i]
        NewArrayOfClasses[ListOfIndices[i]] = ArrayOfOnehot[i]
#         NewListOnehot[ListOfIndices[i]] = ListOnehot[i]
#         NewListVectors[ListOfIndices[i]] = ListVectors[i]
        
        
    # Split Set into train (70%) and test (30%) sets
    # Get number of reads in train set and test sets
    NumReadsTrain = int(math.ceil(0.7 * len(ArrayOfVectors)))
#     NewArrayOfVectors = np.asarray(NewListVectors)
#     NewArrayOfClasses = np.asarray(NewListVectors)
    NewArrayOfVectors = NewArrayOfVectors.astype(float)
    NewArrayOfClasses = NewArrayOfClasses.astype(float)
    X_train = NewArrayOfVectors[:NumReadsTrain]
    Y_train = NewArrayOfClasses[:NumReadsTrain]
    X_test = NewArrayOfVectors[NumReadsTrain:]
    Y_test = NewArrayOfClasses[NumReadsTrain:]
    print('Number of reads in Train set: {}'.format(NumReadsTrain))
    print('Number of reads in Test set: {}'.format(len(ArrayOfVectors) - NumReadsTrain))
    print('Size of X_train: {}'.format(len(X_train)))
    print('Size of Y_train: {}'.format(len(Y_train)))
    print('Size of X_test: {}'.format(len(X_test)))
    print('Size of Y_test: {}'.format(len(Y_test)))
    print(X_train.shape)
    print(Y_train.shape)
    return X_train, Y_train, X_test, Y_test




# Function that returns a dictionary of all possible kmers with 
# each kmer associated with an integer
def GetKmersDictionary(k_value=10):
    # Create empty list to store all possible kmers
    list_kmers = []
    list_nt = ['A', 'T', 'C', 'G']
    # Get list of all possible 4**k_value kmers
    get_all_kmers(list_nt, "", len(list_nt), k_value, list_kmers)
    # generate a list of integers
    list_num = list(range(0,len(list_kmers)))
    # Assign an integer to each kmer
    kmers_dict = dict(zip(list_kmers, list_num))
    return kmers_dict


# In[ ]:


# Function to create a list of all possible kmers
def get_all_kmers(list_nt, prefix, n, k, list_kmers):
    if k == 0 :
        list_kmers.append(prefix)
        return list_kmers
    
    for i in range(n):
        newPrefix = prefix + list_nt[i]
        get_all_kmers(list_nt, newPrefix, n, k-1, list_kmers)


# In[5]:


# Function that looks for any characters different 
# from A, C, T or G and converts the DNA sequence into a vector of 10-mers 
def ParseSeq(DNAsequence,kmers_dict):
    # create empty list to store all the kmers
    listKmers = []
    
    # Drop the read if it contains letters other than A, C, T or G
    if not re.match('^[ATCG]+$', DNAsequence):
        return listKmers

    # Creates a sliding window of width 10
    for n in range(len(DNAsequence) - 9):
        kmer = DNAsequence[n:n+10]
        # Lookup integer mapped to the kmer
        kmer_Integer = kmers_dict[kmer]
        # Add kmer to vector of kmer
        listKmers.append(kmer_Integer)

    # Pad or truncate list to 141 kmers
    listKmers = listKmers[:141] + [0]*(141 - len(listKmers))
    # transform list into numpy array
    array_int = np.asarray(listKmers)
    # Flip array
    array_int = np.flip(array_int, axis=0)
    if len(array_int) != 141:
        print('Size of kmer array: {}'.format(len(array_int)))
       	print('DNA sequence: {}'.format(DNAsequence))
    return array_int   


# In[ ]:


# Function that parse the Fastq Files and process the reads 
def ParseFastq(ListGenomes, DictClasses, kmers_dict, class_mapping):
    # initialize 2 arrays to store the integer vectors and the classes integer
    ArrayOfVectors = []
    ArrayOfClasses = []
    ArrayOfIDs = []
    for FastqFile in ListGenomes:
        print(datetime.datetime.now(), file=sys.stderr)
        print(FastqFile, file=sys.stderr)
        print(len(ArrayOfVectors), file=sys.stderr)
        CompleteFastqFile = DictClasses[FastqFile] + '/' + FastqFile
        # Create a dictionary like object that store information about the reads
        Reads_dict = SeqIO.index(CompleteFastqFile, 'fastq')
        total_number_reads = 0
        for recordID in Reads_dict.keys():
            # Check read sequence
            seq_record = Reads_dict[recordID]
            KmerVector = ParseSeq(str(seq_record.seq), kmers_dict)
            if len(KmerVector) == 141:
                # Add read to array of reads
#                 ArrayOfVectors = UpdateVectorArray(ArrayOfVectors, KmerVector)
                ArrayOfVectors.append(KmerVector)
                ArrayOfIDs.append(seq_record.id)
                # Get Class integer
                for ClassInteger, ClassName in class_mapping.items():
                    if ClassName == DictClasses[FastqFile]: 
                        # Add Class of read to array of classes
                        ArrayOfClasses.append(ClassInteger)
#                         ArrayOfClasses = UpdateIntegerArray(ArrayOfClasses, ClassInteger)           
            total_number_reads += 1
        print('Total number of reads in {0}: {1}'.format(FastqFile, total_number_reads), file=sys.stderr)
    print('Total number of reads in Dataset: {}'.format(len(ArrayOfVectors)), file=sys.stderr)
    return np.asarray(ArrayOfVectors), np.asarray(ArrayOfClasses), np.asarray(ArrayOfIDs)


# In[ ]:


# Function that creates the dataset of simulated reads from all the fastq files available
def GetSetInfo(File):
    ListGenomes = []
    DictClasses = {}
    with open(File, 'r') as Classes:
        for IndClass in Classes:
            IndClass = str(IndClass.strip('\n'))
            # Get list of Fastq files for the given Class
            FastqFiles = open(IndClass + '/Genomes.txt', 'r')
            ListFastqFiles = []
            for line in FastqFiles:
                line = str(line.strip('\n'))
                ListFastqFiles.append(line)
            # Choose 8 files randomly
            random.shuffle(ListFastqFiles)
            # Add directory to file name
            ListGenomes += ListFastqFiles
#            ListGenomes += ListFastqFiles[:5]
            # Rewrite file of Fastq Files names with new set of Genomes
#            with open(IndClass + '/Genomes.txt', 'w') as NewFastqFiles:
#                for File in ListFastqFiles[5:]:
#                    NewFastqFiles.write(File + '\n')
            # Add fastq files to DictClasses
#            for j in range(len(ListFastqFiles[:5])):
#                DictClasses[ListFastqFiles[:5][j]] = IndClass
            for j in range(len(ListFastqFiles)):
                DictClasses[ListFastqFiles[j]] = IndClass
    print('DictClasses: {}'.format(DictClasses))
    print('List Genomes: {}'.format(ListGenomes))
    return ListGenomes, DictClasses


# In[ ]:


def DropReads(X_set, Y_set, Global_size_batch):
    if len(X_set) % Global_size_batch != 0: 
        # Get Number of batches of size Global_size_batch
        NumBatches = int(len(X_set) / Global_size_batch)
        # Get number of reads to remove
        NumReadsToRemove = len(X_set) - (Global_size_batch * NumBatches)
        print('Number of reads to remove: {}'.format(NumReadsToRemove))
        # Drop reads from vectors
        X_set = X_set[:-NumReadsToRemove]
        Y_set = Y_set[:-NumReadsToRemove]
    return X_set, Y_set

def GetElapsedTime(start,end):
    elapsedTime = end - start
    elapsedTimeInSeconds = elapsedTime.total_seconds()
    hours = divmod(elapsedTimeInSeconds, 3600)[0]
    minutes = divmod(elapsedTimeInSeconds, 60)[0]
    seconds = elapsedTime.seconds
    print("Elapsed time: {0} hours, {1} minutes and {2} seconds".format(hours,minutes,seconds))

def main():
    #### SET MODEL PARAMETERS #####
    embedding_size = 6
    num_epochs = 2
    k_value = 10
    num_kmers = 4**k_value
    num_classes = 8
    hidden_size = 512
    # Set size of vector representing each read
    sequence_length = int(150 - k_value + 1)

    ######## GET DATA #################
    # Get dictionary mapping all possible 10-mers to integers
    kmers_dict = GetKmersDictionary()
    print('Number of {0}-mers: {1}'.format(k_value,len(kmers_dict)))
    print('Integer mapped to AAAAAAAAAA: {}'.format(kmers_dict['AAAAAAAAAA']))
    # create dictionary mapping each class to an integer
    class_mapping = {0: 'Enterobacterales', 1: 'Mycoplasmatales', 2: 'Chlamydiales', 3: 'Vibrionales',
                     4: 'Fusobacteriales', 5: 'Spirochaetales', 6: 'Rhodobacterales', 7: 'Unclassified'}
    # Get File with genomes IDs
    InfoDataset = str(sys.argv[1])
    # get list of genomes in file directory to reads
    ListGenomes, DictClasses = GetSetInfo(InfoDataset)
    print(len(ListGenomes))
    print(len(DictClasses))
    print("Number of Genomes in Dataset: {}".format(len(ListGenomes)), file=sys.stderr)
    print("\nProcess reads: START -- {}".format(datetime.datetime.now()),file=sys.stderr)
    start_process_reads = datetime.datetime.now()
    # Convert each read as an integer array
    ArrayOfVectors, ArrayOfClasses, ArrayOfIDs = ParseFastq(ListGenomes, DictClasses, kmers_dict, class_mapping)
    # Create matrix of vectors
    X_train, Y_train, X_test, Y_test = CreateSets(ArrayOfVectors, ArrayOfClasses, ArrayOfIDs, sequence_length, num_classes)
    print("\nProcess reads: END -- {}".format(datetime.datetime.now()), file=sys.stderr)
    end_process_reads = datetime.datetime.now()
    print("Time for reads processing:")
    GetElapsedTime(start_process_reads, end_process_reads)
  
    ########### TRAINING + TESTING ###############
    
    # create a strategy to distribute the variables and the graph across multiple GPUs
    strategy = tf.distribute.MirroredStrategy()
#    strategy = tf.distribute.OneDeviceStrategy(device="/XLA_GPU:0")
    print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    
    # Set batch size
    batch_size_per_device = 64
    global_batch_size = batch_size_per_device * strategy.num_replicas_in_sync

    # Have a number of reads in the train and test sets that is a multiple of the number of devices available
    X_train, Y_train = DropReads(X_train,Y_train,global_batch_size)
    print('Size of X_train: {}'.format(len(X_train)))
    print('Size of Y_train: {}'.format(len(Y_train)))
    X_test, Y_test = DropReads(X_test,Y_test,global_batch_size)
    print('Size of X_test: {}'.format(len(X_test)))
    print('Size of Y_test: {}'.format(len(Y_test)))

    # Set size to shuffle dataset
    BUFFER_SIZE = len(X_train)
    with strategy.scope():
#         Define model
#         Model created with the keras functional API
#         Input layer receives vectors of 141 integers, between 0 and 4**k_value
#         RNN_inputs = tf.keras.Input(shape=(141,), name='input_layer')
#         Define embedding layer that encodes the input vectors into a sequence of dense 6-dimensional vectors
#         kmers_embedded = tf.keras.layers.Embedding(input_dim=num_kmers,output_dim=embedding_size,input_length=sequence_length)(RNN_inputs)  
#         Use the CuDNN kernel
#         hidden1 = tf.keras.layers.LSTM(hidden_size)(kmers_embedded)
#        hidden1 = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(hidden_size,activation='tanh',dropout=0.5))(kmers_embedded)
#        hidden2 = tf.keras.layers.Dense(hidden_size,activation='relu')(hidden1)
#         RNN_outputs = tf.keras.layers.Dense(num_classes,activation='softmax')(hidden1)
#         model = tf.keras.Model(inputs=RNN_inputs, outputs=RNN_outputs)
#         print(model.layers[0].get_weights())
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(input_dim=num_kmers,output_dim=embedding_size,input_length=sequence_length))
        model.add(tf.keras.layers.LSTM(hidden_size))
        model.add(tf.keras.layers.Dense(num_classes,activation='softmax'))
        
        print(model.summary())
        with open('modelsummary.txt', 'w') as f:
            with redirect_stdout(f):
                model.summary()
        # plot graph
        tf.keras.utils.plot_model(model, to_file='recurrent_neural_network.png')
        # Create trainer object
        trainer = MultiGPUs(num_epochs, model, batch_size_per_device, strategy)
        # Slice and batch train and test datasets
        Train_dataset_iter = tf.data.Dataset.from_tensor_slices((X_train,Y_train)).shuffle(BUFFER_SIZE).batch(global_batch_size)
        Test_dataset_iter = tf.data.Dataset.from_tensor_slices((X_test,Y_test)).batch(global_batch_size)
        # Distribute train and test datasets
        train_dist_dataset = strategy.experimental_distribute_dataset(Train_dataset_iter)
        test_dist_dataset = strategy.experimental_distribute_dataset(Test_dataset_iter)
        # Get number of elements in train and test datasets
        num_elements_train_dist_dataset = 0
        for inputs in train_dist_dataset:
           num_elements_train_dist_dataset += 1
        print('Number of elements in distributed train dataset: {}'.format(num_elements_train_dist_dataset))
        # Train model
        start_training_time = datetime.datetime.now()
        print("\nStart training: {}".format(datetime.datetime.now()), file=sys.stderr)
        trainer.custom_training_loop(train_dist_dataset, test_dist_dataset, strategy)
        print("\nEnd training: {}".format(datetime.datetime.now()), file=sys.stderr)
        end_training_time = datetime.datetime.now()
        print("Time for training:")
        GetElapsedTime(start_training_time, end_training_time)
        print(model.layers[0].get_weights())
        # delete TF graph and create a new one
#        tf.keras.backend.clear_session()
# In[ ]:


if __name__ == "__main__":
    main()

