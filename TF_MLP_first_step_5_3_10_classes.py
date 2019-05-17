
# coding: utf-8

# In[1]:


# Implementation of an artificial neural network
# Load all necessary libraries
from __future__ import division, print_function
import sys
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
import statistics
from keras import regularizers
from operator import add


# In[22]:


# Load the data
mtx_df = pd.read_csv("/Users/Cissou/Desktop/matrix_wo_zero.csv", sep='\t')


# In[23]:


# Function that will round up a number to a specified number of digits
def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier


# In[24]:


# Normalize the data using min max normalization (keep same dataframe name)
# Create new empty dataframe to store the data normalized and keep
# only the rows of genomes belonging to the top 10 orders
headers = mtx_df.columns.values.tolist()
num_rows, num_col = mtx_df.shape
mtx_df[headers[9:len(headers)+1]] = np.round((mtx_df[headers[9:len(headers)+1]]-mtx_df[headers[9:len(headers)+1]].min())/(mtx_df[headers[9:len(headers)+1]].max()-mtx_df[headers[9:len(headers)+1]].min()), 3)
print(mtx_df)


# In[25]:


print(mtx_df['order'])
n_features = 1024
n_classes = 10
reg_lambda = 0.01
size_batch = 64
num_hidden_units = 50
num_epochs = 1000
eta = 0.002


# In[26]:


# Count the number of genomes per order and retain only the top 10 orders with the highest number of genomes
orders_to_keep = mtx_df['order'].value_counts().head(10).index.tolist()
# Get the count for each species in dataset
species_list_count = mtx_df['species'].value_counts().to_dict()
# iterate through column 'order' and get the rows in dataframe of orders that are not in list orders_to_keep
list_index_to_keep = []
# Get the species that have more than 4 genomes
species_to_keep = []
species_to_remove = {} # key=order, v = dictionary with key= species, value=# genomes
list_index_species_removed = []
for j in range(mtx_df.shape[0]):
    if mtx_df.iloc[j,4] in orders_to_keep:
        if species_list_count[mtx_df.iloc[j,7]] >= 4:
            species_to_keep.append(mtx_df.iloc[j,7])
            list_index_to_keep.append(j)
        else:
            if mtx_df.iloc[j,4] not in species_to_remove:
                species_to_remove[mtx_df.iloc[j,4]] = {}
                species_to_remove[mtx_df.iloc[j,4]][mtx_df.iloc[j,7]] = species_list_count[mtx_df.iloc[j,7]]
            else:
                if mtx_df.iloc[j,7] not in species_to_remove[mtx_df.iloc[j,4]]:
                    species_to_remove[mtx_df.iloc[j,4]][mtx_df.iloc[j,7]] = species_list_count[mtx_df.iloc[j,7]]
            list_index_species_removed.append(j)


# In[31]:


# Encode class labels as integers
# Enumerate the class labels starting at 0
class_mapping = {label:idx for idx, label in enumerate(np.unique(mtx_df['order']))}
print(class_mapping)
# Use the mapping dictionary to transform the class labels into integers
mtx_df['order'] = mtx_df['order'].map(class_mapping)
#print(class_mapping)
#print(mtx_df['order'])
indices = [0, 1, 2]
depth = 3
  # output: [3 x 3]
# [[1., 0., 0.],
#  [0., 1., 0.],
#  [0., 0., 1.]]
vector = tf.one_hot(indices, depth)
print(vector)
with tf.Session():
    Tensor.eval(vector)


# In[9]:


# dictionary with order for each species, key = species, value = order in resulting dataframe
species_orders_dict = {} #keys = species, values = order
species_indices_dict = {} #keys = species, values = list of indices
#species_genuses_dict = {} #keys = genus, values = list of species
#list_indices_mtx_df = mtx_df.index.values.tolist()
for j in range(len(list_index_to_keep)):
    if mtx_df.loc[list_index_to_keep[j],'species'] not in species_orders_dict:
        species_orders_dict[mtx_df.loc[list_index_to_keep[j],'species']] = mtx_df.loc[list_index_to_keep[j],'order']
    if mtx_df.loc[list_index_to_keep[j],'species'] not in species_indices_dict:
        species_indices_dict[mtx_df.loc[list_index_to_keep[j],'species']] = []
        species_indices_dict[mtx_df.loc[list_index_to_keep[j],'species']].append(list_index_to_keep[j])
    else:
        species_indices_dict[mtx_df.loc[list_index_to_keep[j],'species']].append(list_index_to_keep[j])
#    if mtx_df.loc[list_index_to_keep[j],'genus'] not in species_genuses_dict:
#        species_genuses_dict[mtx_df.loc[list_index_to_keep[j],'genus']] = []
#        species_genuses_dict[mtx_df.loc[list_index_to_keep[j],'genus']].append(mtx_df.loc[list_index_to_keep[j],'species'])
#    else:
#        if mtx_df.loc[list_index_to_keep[j],'species'] not in species_genuses_dict[mtx_df.loc[list_index_to_keep[j],'genus']]:
#            species_genuses_dict[mtx_df.loc[list_index_to_keep[j],'genus']].append(mtx_df.loc[list_index_to_keep[j],'species'])  
num_genomes = 0
for k, v in species_indices_dict.items():
    num_genomes += len(v)
print(num_genomes)


# In[10]:


# Create entire training and test sets
train_idx = []
test_idx = []
# For each species, select 70% of indices for training set and the remaining 30% for test set
for k, v in species_indices_dict.items():
    # shuffle list
    random.shuffle(v)
    # select 70% of indices for the training set 
    num_genomes_train = int(0.7*len(v))
    train_idx += v[:num_genomes_train]
    test_idx += v[num_genomes_train:]


# In[12]:


# Create 2 dataframes to keep the data for training and test sets
train_df = mtx_df.loc[train_idx,:]
test_df = mtx_df.loc[test_idx,:]
print(train_df.shape)
print(test_df.shape)
columns_to_keep = [headers[4]] + headers[349:1373]
training_set = train_df.loc[:,columns_to_keep]
training_set.to_csv('/Users/Cissou/Desktop/Training_set.csv', sep='\t',index=False)
testing_set = test_df.loc[:,columns_to_keep]
testing_set.to_csv('/Users/Cissou/Desktop/Testing_set.csv', sep='\t',index=False)

#Split the features and the labels for both the training and testing sets 
# and only select the order and the 5-mers columns
X_train_all = train_df[headers[349:1373]].values
Y_train_all = train_df['order'].values
X_test_all = test_df[headers[349:1373]].values
Y_test_all = test_df['order'].values
print(Y_test_all)


# In[40]:


# Create new empty computation graph
g = tf.Graph()


# In[41]:


# Define the model
with g.as_default(): 
    # Set a graph-level seed to make the random sequences generated by all operations be repeatable across sessions
    tf.set_random_seed(1234)
    # Create predefined tensors (placeholders) for holding the features values (tf_x) and the response values (tf_y)
    tf_x = tf.placeholder(dtype=tf.float32, shape=(None, n_features), name='tf_x')
    tf_y = tf.placeholder(dtype=tf.int32, shape=None, name='tf_y')
    # Convert labels to a one-hot representation (necessary to use the cost function)
    y_onehot = tf.one_hot(indices=tf_y, depth=n_classes)
    # Define properties of the first layer h0 or input layer with L2 regularization
    h1 = tf.layers.dense(inputs=tf_x, units=num_hidden_units, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(reg_lambda), name='layer1')
    # Define properties of the output layer logits = vector of raw (non-normalized) predictions
    logits = tf.layers.dense(inputs=h1, units=n_classes, activation=None, name='layer2')
    predictions = {
        # argmax: Returns the index with the largest value across axes of a tensor.
        'classes': tf.argmax(logits, axis=1, name='predicted_classes'),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    index = tf.argmax(one_hot_vector, axis=0)
    # Compute the cost
    cost = tf.losses.softmax_cross_entropy(onehot_labels=y_onehot, logits=logits)
    # Construct a new gradient descent optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=eta)
    print(eta)
    train_op = optimizer.minimize(loss=cost)
    # Returns an Operation that initializes global variables in the graph
    init_op = tf.global_variables_initializer()


# In[42]:


# Define function that generates batches of data
def create_batch_generator(x, y, batch_size=128, shuffle=False):
    x_copy = np.array(x)
    y_copy = np.array(y)
    if shuffle:
        data = np.column_stack((x_copy, y_copy))
        np.random.shuffle(data)
        x_copy = data[:, :-1]
        y_copy = data[:, -1].astype(int)
    for i in range(0, x.shape[0], batch_size):
        yield (x_copy[i:i+batch_size, :], y_copy[i:i+batch_size])


# In[44]:


# Define run_train function
def run_train(sess, X_train, Y_train):
    print("\nStart training")
    # initialize all variables
    sess.run(init_op)
    # create a list to maintain average training costs for each epoch
    plot_values = []
    for epoch in range(num_epochs):
        training_costs = []
        batch_generator = create_batch_generator(X_train, Y_train, batch_size=64)
        for batch_X, batch_Y in batch_generator:
            # prepare a dict to feed data to our network
            feed = {tf_x:batch_X, tf_y:batch_Y}
            # train the network
            _, batch_cost = sess.run([train_op, cost], feed_dict=feed)
            training_costs.append(batch_cost)
            #print(' -- Epoch %2d ''Avg. Training Loss: %.4f' % (epoch+1, np.mean(training_costs)))
        plot_values.append(np.mean(training_costs))
    # Generate a plot to visualize the training costs
    plt.clf()
    plt.plot(range(1,len(plot_values) + 1), plot_values)
    plt.xlabel('Epoch')
    plt.ylabel('Training Cost')
    plt.savefig('Loss_{0}.png'.format(eta))  


# In[ ]:


# Create a new TF session to launch the graph g
with tf.Session(graph=g) as sess1:
    # train model with all training data
    run_train(sess1, X_train_all, Y_train_all)
    # Use the trained model to do predictions on the test dataset
    y_pred = sess1.run(predictions['classes'], feed_dict={tf_x: X_test_all})
    # Get unique 
    y_pred_un = np.unique(y_pred)
    # Create a confusion matrix
    cm = pd.DataFrame(0, columns=list(np.unique(Y_test_all)), index=list(np.unique(Y_test_all)))
    # Fill cm
    for i in range(len(Y_test_all)):
        cm.iloc[cm.index.get_loc(Y_test_all[i]),cm.columns.get_loc(y_pred[i])] += 1
    print(cm)
    # compute recall, precision for test set
    recall = cm.iloc[1,1] / (cm.iloc[1,0] + cm.iloc[1,1])
    precision = cm.iloc[1,1] / (cm.iloc[1,1] + cm.iloc[0,1])
    print('Test Accuracy entire test set: %.2f%%' % (100*np.sum(y_pred == Y_test_all)/Y_test_all.shape[0]))
    print('Test Recall Order 1 entire test set: %.2f%%' % (100*recall))
    print('Test Precision Order 1 entire test set: %.2f%%' % (100*precision))

