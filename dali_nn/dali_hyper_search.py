import nvidia.dali as dali
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.plugin.tf as dali_tf
import nvidia.dali.tfrecord as tfrec

import tensorflow as tf
from tensorflow import keras
from tensorboard.plugins.hparams import api as hp
from keras.callbacks import TensorBoard

from glob import glob
import argparse
import numpy as np
import json
from collections import defaultdict
import os
import sys
import io
import math
import datetime

from summarize import learning_curves

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
print(f'TF version: {tf.__version__}')
print(f'python version: {sys.version}')
sys_details = tf.sysconfig.get_build_info()
cuda_version = sys_details["cuda_version"]
print(f'CUDA version: {cuda_version}')
cudnn_version = sys_details["cudnn_version"]
print(f'CUDNN version: {cudnn_version}')
print(sys_details)
print("NCCL DEBUG SET")
os.environ["NCCL_DEBUG"] = "WARN"

# setup hyperparameters to experiment
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([32, 64, 128, 256]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.5, 0.6))
HP_EMBEDDING_SIZE = hp.HParam('embedding_size', hp.Discrete([20, 60, 100]))
#HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd', 'rmsprop']))

METRIC_ACCURACY = 'accuracy'

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
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


class CreateCheckpoints(tf.keras.callbacks.Callback):
    """ save model at the end of every epoch """
    def __init__(self, ckpts_dir):
        super(CreateCheckpoints, self).__init__()
        self.ckpts_dir = ckpts_dir

    def on_epoch_end(self, epoch, logs=None):
        #if epoch % 5 == 0 or epoch + 1 == EPOCHS:
        self.model.save(self.ckpts_dir + f'epoch-{epoch}')

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

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get("val_loss")
        # save checkpoints
        self.model.save(os.path.join(input_path, 'ckpts', f'ckpts-epoch{epoch}'))
        if current_val_loss < self.best:
            self.best = current_val_loss
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)
                self.model.save(os.path.join(input_path, 'ckpts', f'ckpts-epoch{epoch}-best'))

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

def scheduler(epoch, lr):
    """ learning rate scheduler """
    if epoch % 10 == 0 and epoch != 0:
        print(f'epoch when changing lr: {epoch}')
        lr = lr / 2
        return lr
    else:
        return lr

class TFRecordPipeline(Pipeline):
    def __init__(self, batch_size, tfrecord, tfrecord_idx, device_id=0, shard_id=0, num_shards=1, num_threads=4,
                 seed=0):
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


def train_test_model(args, ckpts_dir, lc_filename, hparams, strategy):
    # Define model to train: AlexNet
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(args.vector_size), dtype='int32'),
            tf.keras.layers.Embedding(input_dim=args.vocab_size + 1, output_dim=hparams[HP_EMBEDDING_SIZE],
                                      input_length=args.vector_size, mask_zero=True, trainable=True),
            tf.keras.layers.Reshape((args.vector_size, hparams[HP_EMBEDDING_SIZE], 1)),
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
            tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
            tf.keras.layers.Dense(units=4096),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
            tf.keras.layers.Dense(units=1000),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
            tf.keras.layers.Dense(args.num_classes),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('softmax'),
        ]
    )

    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'],
                  optimizer='adam')

    # define shapes and types of data and labels
    shapes = (
        (hparams[HP_BATCH_SIZE], args.vector_size),
        (hparams[HP_BATCH_SIZE]))
    dtypes = (
        tf.int64,
        tf.int64)

    # distribute dataset
    def train_dataset_fn(input_context):
        with tf.device("/gpu:{}".format(input_context.input_pipeline_id)):
            device_id = input_context.input_pipeline_id
            return dali_tf.DALIDataset(fail_on_device_mismatch=False,
                                       pipeline=TFRecordPipeline(hparams[HP_BATCH_SIZE], args.train_tfrecord, args.train_tfrecord_idx,
                                                                 device_id=device_id, shard_id=device_id,
                                                                 num_shards=args.gpus),
                                       batch_size=hparams[HP_BATCH_SIZE], output_shapes=shapes, output_dtypes=dtypes,
                                       device_id=device_id)

    def val_dataset_fn(input_context):
        with tf.device("/gpu:{}".format(input_context.input_pipeline_id)):
            device_id = input_context.input_pipeline_id
            return dali_tf.DALIDataset(fail_on_device_mismatch=False,
                                       pipeline=TFRecordPipeline(hparams[HP_BATCH_SIZE], args.val_tfrecord, args.val_tfrecord_idx,
                                                                 device_id=device_id, shard_id=device_id,
                                                                 num_shards=args.gpus),
                                       batch_size=hparams[HP_BATCH_SIZE], output_shapes=shapes, output_dtypes=dtypes,
                                       device_id=device_id)

    # get distributed datasets
    input_options = tf.distribute.InputOptions(
        experimental_place_dataset_on_device=True,
        experimental_prefetch_to_device=False,
        experimental_replication_mode=tf.distribute.InputReplicationMode.PER_REPLICA)

    train_dataset = strategy.distribute_datasets_from_function(train_dataset_fn, input_options)
    val_dataset = strategy.distribute_datasets_from_function(val_dataset_fn, input_options)

    if args.early_stopping == 'true':
        history = model.fit(train_dataset, epochs=args.epochs, steps_per_epoch=args.validation_steps, verbose=2,
                                callbacks=[tf.keras.callbacks.LearningRateScheduler(scheduler), CreateCheckpoints(ckpts_dir),
                                           hp.KerasCallback(os.path.join(args.full_input_path, f'logs-fold{args.fold}-hparams'), hparams), EarlyStoppingAtMinLoss()])

    elif args.early_stopping == 'false':
        history = model.fit(train_dataset, epochs=args.epochs, steps_per_epoch=args.training_steps, verbose=2,
                                callbacks=[CreateCheckpoints(ckpts_dir),tf.keras.callbacks.LearningRateScheduler(scheduler),
                                  hp.KerasCallback(os.path.join(args.full_input_path, f'logs-fold{args.fold}-hparams'), hparams)])

    _, accuracy = model.evaluate(val_dataset, validation_steps=args.validation_steps)

    # create learning curves
    learning_curves(history, lc_filename)

    return accuracy


def run(args, run_dir, ckpts_dir, lc_filename, hparams, strategy):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy = train_test_model(args, ckpts_dir, lc_filename, hparams, strategy)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help="Path to input directory to store results", required=True)
    parser.add_argument('--type', type=str, help="cross-validation or training with complete data", choices=['cv', 'standard'], default='standard')
    parser.add_argument('--fold', type=int, help="fold of dataset if in cross validation mode", required=('cv' in sys.argv))
    parser.add_argument('--epochs', type=int, help="number of epochs of training", default=50)
    parser.add_argument('--early_stopping', type=str, help="implement early stopping or not", choices=['true', 'false'], default='false')
    parser.add_argument('--gpus', type=int, help="number of gpus", default=2)
    parser.add_argument('--training_size', type=int, help="number of reads in training set", required=True)
    parser.add_argument('--validation_size', type=int, help="number of reads in validation set", required=True)

    args = parser.parse_args()

    #get number of hyperparameters search runs
    model_num = 1 + len(glob(os.path.join(args.input_path, 'hp*')))
    args.full_input_path = os.path.join(args.input_path, f'hp{model_num}')
    if not os.path.exists(args.full_input_path):
        os.makedirs(args.full_input_path)

    f = open(os.path.join(args.input_path, 'class_mapping.json'))
    class_mapping = json.load(f)

    K_VALUE = 12
    READ_LENGTH = 250
    args.num_classes = len(class_mapping)
    NUM_DEVICES = args.gpus  # number of GPUs
    args.vocab_size = 8390658
    args.vector_size = READ_LENGTH - K_VALUE + 1

    args.train_tfrecord = os.path.join(args.input_path, 'tfrecords', f'training_data_fold{args.fold}.tfrec')
    args.train_tfrecord_idx = os.path.join(args.input_path, 'tfrecords', f'idx_files/training_data_fold{args.fold}.tfrec.idx')
    args.val_tfrecord = os.path.join(args.input_path, 'tfrecords', f'validation_data_fold{args.fold}.tfrec')
    args.val_tfrecord_idx = os.path.join(args.input_path, 'tfrecords', f'idx_files/validation_data_fold{args.fold}.tfrec.idx')

    # write down settings for training
    f = open(os.path.join(args.full_input_path, f'HP-info-cv-fold{args.fold}'), 'w')
    f.write(f'reads in training set: {args.training_size}\nreads in validation set: {args.validation_size}\n'
        f'read length: {READ_LENGTH}\nk value: {K_VALUE}\nnumber of devices: {NUM_DEVICES}\nnumber of classes: {args.num_classes}\n'
        f'number of kmers: {args.vocab_size}\ninput size: {args.vector_size}\n')

    with tf.summary.create_file_writer(os.path.join(args.full_input_path, 'logs/hparam_tuning')).as_default():
        hp.hparams_config(
            hparams=[HP_DROPOUT, HP_EMBEDDING_SIZE, HP_BATCH_SIZE],
            metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
        )   

    print("Fit model on training data")
    # create an instance of strategy to perform synchronous training across multiple gpus
    strategy = tf.distribute.MirroredStrategy(devices=['/gpu:0', '/gpu:1'])
    #strategy = tf.distribute.MirroredStrategy(devices=['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3'])
    session_num = 0
    with strategy.scope():
        for batch_size in HP_BATCH_SIZE.domain.values:
            for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
                for optimizer in HP_EMBEDDING_SIZE.domain.values:
                    hparams  = {
                        HP_BATCH_SIZE: batch_size,
                        HP_DROPOUT: dropout_rate,
                        HP_EMBEDDING_SIZE: optimizer
                    }
                    tf.compat.v1.reset_default_graph()
                    start = datetime.datetime.now()
                    run_name = "run-%d" % session_num

                    CKPTS_DIR = os.path.join(args.full_input_path, f'{run_name}', 'ckpts')
                    LC_FILENAME = os.path.join(args.full_input_path, f'{run_name}', f'LearningCurves.png')
                    
                    GLOBAL_BATCH_SIZE = NUM_DEVICES * hparams[HP_BATCH_SIZE]
                    args.training_steps = math.ceil(args.training_size / GLOBAL_BATCH_SIZE)
                    args.validation_steps = math.ceil(args.validation_size / GLOBAL_BATCH_SIZE)

                    f.write(f'Run {run_name}\n')
                    for h in hparams:
                        f.write(f'{h.name}:\t{hparams[h]}\n')
                    f.write(f'global batch size: {GLOBAL_BATCH_SIZE}\n'
                            f'training steps: {args.training_steps}\n'
                            f'validation steps: {args.validation_steps}\n')
                    
                    print(f'Starting trial: {run_name}')
                    print({h.name: hparams[h] for h in hparams})
                    run(args, os.path.join(args.full_input_path, f'logs-fold{args.fold}/hparam_tuning', run_name), CKPTS_DIR, LC_FILENAME, hparams, strategy)
                    end = datetime.datetime.now()
                    total_time = end - start
                    hours, seconds = divmod(total_time.seconds, 3600)
                    minutes, seconds = divmod(seconds, 60)
                    print("\nTook %02d:%02d:%02d.%d\n" % (hours, minutes, seconds, total_time.microseconds))
                    f.write(f'runtime: {hours}:{minutes}:{seconds}.{total_time.microseconds}\n')
    f.close()



