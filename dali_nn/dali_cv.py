import nvidia.dali as dali
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.plugin.tf as dali_tf
import nvidia.dali.tfrecord as tfrec

import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.callbacks import TensorBoard

import numpy as np
import json
from collections import defaultdict
import os
import sys
import io
import math
import datetime

from summarize import learning_curves, create_barplot_training

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
print(f'TF version: {tf.__version__}')
print(f'python version: {sys.version}')
sys_details = tf.sysconfig.get_build_info()
cuda_version = sys_details["cuda_version"]
print(f'CUDA version: {cuda_version}')
cudnn_version = sys_details["cudnn_version"]
print(f'CUDNN version: {cudnn_version}')
print(sys_details)
#print("NCCL DEBUG SET")
#os.environ["NCCL_DEBUG"] = "WARN"

tf.compat.v1.reset_default_graph()


FOLD = int(sys.argv[1])
input_path = str(sys.argv[2])
tfrecords_path = str(sys.argv[3])
#model = str(sys.argv[4])
#last_epoch = int(model.split('/')[-1].split('-')[2]) + 1
#print(f'resume training of model at epoch: {last_epoch}')
last_epoch = 1

f = open(os.path.join(input_path, 'class_mapping.json'))
class_mapping = json.load(f)

K_VALUE = 12
READ_LENGTH = 250
#WINDOW_SIZE = 5
NUM_CLASSES = len(class_mapping)
NUM_DEVICES = 4      # number of GPUs
BATCH_SIZE = 250  # batch size per GPU
GLOBAL_BATCH_SIZE = NUM_DEVICES*BATCH_SIZE
VOCAB_SIZE = 8390658
EPOCHS = 1
EMBEDDING_SIZE = 60
DROPOUT_RATE = 0.5
#NUM_NS = 4
VECTOR_SIZE = READ_LENGTH - K_VALUE + 1
TRAINING_SIZE = 5725889
VALIDATION_SIZE = 1431327
#TRAINING_SIZE = 5000
#VALIDATION_SIZE = 1000
STEPS_PER_EPOCH = math.ceil(TRAINING_SIZE / GLOBAL_BATCH_SIZE)
VALIDATION_STEPS = math.ceil(VALIDATION_SIZE / GLOBAL_BATCH_SIZE)

train_tfrecord = os.path.join(tfrecords_path, f'training_data_fold{FOLD}.tfrec')
train_tfrecord_idx = os.path.join(tfrecords_path, f'idx_files/training_data_fold{FOLD}.tfrec.idx')
val_tfrecord = os.path.join(tfrecords_path, f'validation_data_fold{FOLD}.tfrec')
val_tfrecord_idx = os.path.join(tfrecords_path, f'idx_files/validation_data_fold{FOLD}.tfrec.idx')

# write down settings for training
f = open(os.path.join(input_path, f'cross-validation-info-fold{FOLD}'), 'w')
f.write(f'Training of model with fold # {FOLD}\n'
        f'batch size: {BATCH_SIZE}\n'
        f'global batch size: {GLOBAL_BATCH_SIZE}\nsteps per epoch: {STEPS_PER_EPOCH}\nvalidation steps: {VALIDATION_STEPS}\n'
        f'reads in training set: {TRAINING_SIZE}\nreads in validation set: {VALIDATION_SIZE}\n'
        f'read length: {READ_LENGTH}\nk value: {K_VALUE}\nnumber of devices: {NUM_DEVICES}\nnumber of classes: {NUM_CLASSES}\n'
        f'embedding size: {EMBEDDING_SIZE}\ndropout rate: {DROPOUT_RATE}\nnumber of kmers: {VOCAB_SIZE}\ninput size: {VECTOR_SIZE}\n')

# define shapes and types of data and labels
shapes = (
    (BATCH_SIZE, VECTOR_SIZE),
    (BATCH_SIZE))
dtypes = (
    tf.int64,
    tf.int64)

# get list of all visible GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    # enable memory growth to prevent the runtime initialization to allocate all memory on the device
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus, 'GPU')

def dataset_options():
    options = tf.data.Options()
    try:
        options.experimental_optimization.apply_default_optimizations = False
        options.experimental_optimization.autotune = False
    except:
        print('Could not set TF Dataset Options')

    return options



class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir, **kwargs):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': self.model.optimizer._decayed_lr('float32').numpy()})
        lr = self.model.optimizer.lr
        print(f'Learning rate at end of epoch {epoch} is: {self.model.optimizer._decayed_lr("float32").numpy()} - {lr.numpy()}')
        super().on_epoch_end(epoch, logs)


class CreateCheckpoints(keras.callbacks.Callback):
    """ save model at the end of every epoch """
    def __init__(self):
        super(CreateCheckpoints, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0 or epoch + 1 == EPOCHS:
            self.model.save(os.path.join(input_path, 'ckpts', f'ckpts-epoch-{epoch + last_epoch}'))


class EarlyStoppingAtMinLoss(keras.callbacks.Callback):
    """Stop training when the validation loss is at its min, i.e. the validation loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

    def __init__(self, patience=5):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf
        # The epoch with the best weights
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get("val_loss")
        # save checkpoints 
        self.model.save(os.path.join(input_path, 'ckpts', f'ckpts-epoch-{epoch}'))
        if current_val_loss < self.best:
            self.best = current_val_loss
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)
                self.model.save(os.path.join(input_path, 'ckpts', f'ckpts-epoch-{self.best_epoch}-best'))

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

def scheduler(epoch, lr):
    if epoch < 3:
        return lr
    elif epoch >= 3 and epoch < 10:
        return lr * tf.math.exp(-0.1)
    else:
        return lr * tf.math.exp(-0.01)

class TFRecordPipeline(Pipeline):
    def __init__(self, batch_size, tfrecord, tfrecord_idx, device_id=0, shard_id=0, num_shards=1, num_threads=4, seed=0):
        super(TFRecordPipeline, self).__init__(batch_size, num_threads, device_id, seed)
        self.input = ops.TFRecordReader(path=tfrecord, random_shuffle=True, shard_id=shard_id, num_shards=num_shards,
                                        index_path=tfrecord_idx,
                                        features={"read": tfrec.VarLenFeature([], tfrec.int64, 0),
                                                "label": tfrec.FixedLenFeature([1], tfrec.int64, -1)})

    def define_graph(self):
        inputs = self.input()
        reads = inputs["read"]
        labels = inputs["label"]
        labels = labels.gpu()
        return (reads, labels)


# create an instance of strategy to perform synchronous training across multiple gpus
#strategy = tf.distribute.MirroredStrategy(devices=['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3'])
strategy = tf.distribute.MirroredStrategy(devices=['/gpu:0', '/gpu:1'])
with strategy.scope():
    # Define model to train: AlexNet
    # load model
    #model = tf.keras.models.load_model(model)
    #print('model loaded')

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(VECTOR_SIZE), dtype='int32'),
            tf.keras.layers.Embedding(input_dim=VOCAB_SIZE+1, output_dim=EMBEDDING_SIZE,
                                      input_length=VECTOR_SIZE, mask_zero=True, trainable=True),
            tf.keras.layers.Reshape((VECTOR_SIZE, EMBEDDING_SIZE, 1)),
            tf.keras.layers.Conv2D(96, kernel_size=(11, 11), strides=(4, 4), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
            tf.keras.layers.Conv2D(256, kernel_size=(5, 5), strides=(1, 1), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
            tf.keras.layers.Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=4096),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(DROPOUT_RATE),
            tf.keras.layers.Dense(units=4096),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(DROPOUT_RATE),
            tf.keras.layers.Dense(units=1000),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(DROPOUT_RATE),
            tf.keras.layers.Dense(NUM_CLASSES),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('softmax'),
        ]
    )

    model.compile(
       optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
       loss='sparse_categorical_crossentropy',
       metrics=['accuracy'])
   

    with open(os.path.join(input_path, 'model'), 'w+') as model_file:
        model.summary(print_fn=lambda x: model_file.write(x + '\n'))

    # distribute dataset
    def train_dataset_fn(input_context):
        with tf.device("/gpu:{}".format(input_context.input_pipeline_id)):
            device_id = input_context.input_pipeline_id
            return dali_tf.DALIDataset(fail_on_device_mismatch=False,
                pipeline=TFRecordPipeline(BATCH_SIZE, train_tfrecord, train_tfrecord_idx, device_id=device_id, shard_id=device_id, num_shards=NUM_DEVICES),
                batch_size=BATCH_SIZE, output_shapes=shapes, output_dtypes=dtypes, device_id=device_id)

    def val_dataset_fn(input_context):
        with tf.device("/gpu:{}".format(input_context.input_pipeline_id)):
            device_id = input_context.input_pipeline_id
            return dali_tf.DALIDataset(fail_on_device_mismatch=False,
                pipeline=TFRecordPipeline(BATCH_SIZE, val_tfrecord, val_tfrecord_idx, device_id=device_id, shard_id=device_id, num_shards=NUM_DEVICES),
                batch_size=BATCH_SIZE, output_shapes=shapes, output_dtypes=dtypes, device_id=device_id)

    input_options = tf.distribute.InputOptions(
            experimental_prefetch_to_device=False,
            experimental_replication_mode=tf.distribute.InputReplicationMode.PER_REPLICA,
            experimental_place_dataset_on_device=True)

    train_dataset = strategy.distribute_datasets_from_function(train_dataset_fn, input_options)
    val_dataset = strategy.distribute_datasets_from_function(val_dataset_fn, input_options)

    #train_dataset = strategy.experimental_distribute_dataset(train_dataset_fn, input_options)
    #val_dataset = strategy.experimental_distribute_dataset(val_dataset_fn, input_options)

    print("Fit model on training data")
    start = datetime.datetime.now()
    history = model.fit(train_dataset, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, validation_data=val_dataset, validation_steps=VALIDATION_STEPS, callbacks=[tf.keras.callbacks.LearningRateScheduler(scheduler), CreateCheckpoints(), LRTensorBoard(log_dir=os.path.join(input_path, f'logs-fold{FOLD}'))])

    # create learning curves
    learning_curves(history, os.path.join(input_path, f'LearningCurves-fold-{FOLD}.png'))
    
    end = datetime.datetime.now()
    total_time = end - start
    hours, seconds = divmod(total_time.seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    print("\nTook %02d:%02d:%02d.%d\n" % (hours, minutes, seconds, total_time.microseconds))
    f.write(f'runtime: {hours}:{minutes}:{seconds}.{total_time.microseconds}')
    f.close()

    train_true_classes = []
    val_true_classes = []
    def get_labels(inputs, type_set):
        global train_true_classes
        global val_true_classes
        reads, labels = inputs
        t_classes = labels.numpy()
        if type_set == 'train':
            train_true_classes += t_classes.tolist()
        elif type_set == 'val':
            val_true_classes += t_classes.tolist()

    num_steps = 0
    for inputs in train_dataset:
        num_steps += 1
        strategy.run(get_labels, args=(inputs, 'train',))
        if num_steps == STEPS_PER_EPOCH:
            break

    print(f'train: {len(train_true_classes)}')

    num_steps = 0
    for inputs in val_dataset:
        num_steps += 1
        strategy.run(get_labels, args=(inputs, 'val',))
        if num_steps == VALIDATION_STEPS:
            break
    
    print(f'val: {len(val_true_classes)}')

    train_dict_classes = defaultdict(int)
    for j in range(len(train_true_classes)):
        train_dict_classes[train_true_classes[j]] += 1

    val_dict_classes = defaultdict(int)
    for j in range(len(val_true_classes)):
        val_dict_classes[val_true_classes[j]] += 1
    
    create_barplot_training(train_dict_classes, val_dict_classes, os.path.join(input_path, f'data-barplots-fold-{FOLD}'), class_mapping)


