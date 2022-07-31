import tensorflow as tf
import horovod.tensorflow as hvd
import tensorflow.keras as keras
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
import nvidia.dali.plugin.tf as dali_tf
import os
import sys
import json
import glob
import datetime
import numpy as np
import math
import io
import random
import argparse
from collections import defaultdict


# disable eager execution
#tf.compat.v1.disable_eager_execution()
print(f'Is eager execution enabled: {tf.executing_eagerly()}')

# print which unit (CPU/GPU) is used for an operation
#tf.debugging.set_log_device_placement(True)

# enable XLA = XLA (Accelerated Linear Algebra) is a domain-specific compiler for linear algebra that can accelerate
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# Initialize Horovod
hvd.init()
# Map one GPU per process
# use hvd.local_rank() for gpu pinning instead of hvd.rank()
gpus = tf.config.experimental.list_physical_devices('GPU')
print(f'GPU RANK: {hvd.rank()}/{hvd.local_rank()} - LIST GPUs: {gpus}')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

# define the DALI pipeline
@pipeline_def
def get_dali_pipeline(tfrec_filenames, tfrec_idx_filenames, shard_id, num_gpus, dali_cpu=True, training=True):
    inputs = fn.readers.tfrecord(path=tfrec_filenames,
                                 index_path=tfrec_idx_filenames,
                                 random_shuffle=training,
                                 shard_id=shard_id,
                                 num_shards=num_gpus,
                                 initial_fill=10000,
                                 features={
                                     "read": tfrec.VarLenFeature([], tfrec.int64, 0),
                                     "label": tfrec.FixedLenFeature([1], tfrec.int64, -1)})
    # retrieve reads and labels and copy them to the gpus
    reads = inputs["read"].gpu()
    labels = inputs["label"].gpu()
    return (reads, labels)

class DALIPreprocessor(object):
    def __init__(self, filenames, idx_filenames, batch_size, num_threads, vector_size, dali_cpu=True,
               deterministic=False, training=False):

        device_id = hvd.local_rank()
        shard_id = hvd.rank()
        num_gpus = hvd.size()
        self.pipe = get_dali_pipeline(tfrec_filenames=filenames, tfrec_idx_filenames=idx_filenames, batch_size=batch_size,
                                      num_threads=num_threads, device_id=device_id, shard_id=shard_id, num_gpus=num_gpus,
                                      dali_cpu=dali_cpu, training=training, seed=7 * (1 + hvd.rank()) if deterministic else None)

        self.daliop = dali_tf.DALIIterator()

        self.batch_size = batch_size
        self.device_id = device_id

        self.dalidataset = dali_tf.DALIDataset(fail_on_device_mismatch=False, pipeline=self.pipe,
            output_shapes=((batch_size, vector_size), (batch_size)),
            batch_size=batch_size, output_dtypes=(tf.int64, tf.int64), device_id=device_id)

    def get_device_dataset(self):
        return self.dalidataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecords', type=str, help='path to tfrecords')
    parser.add_argument('--idx_files', type=str, help='path to dali index files')
    parser.add_argument('--output_dir', type=str, help='path to store model')
    parser.add_argument('--batch_size', type=int, help='batch size per gpu', default=2048)
    parser.add_argument('--num_train_samples', type=int, help='number of reads in training set')
    parser.add_argument('--num_val_samples', type=int, help='number of reads in validation set')
    args = parser.parse_args()

    VECTOR_SIZE = 250 - 12 + 1

    # Get training and validation tfrecords
    train_files = sorted(glob.glob(os.path.join(args.tfrecords, 'train*.tfrec')))
    train_idx_files = sorted(glob.glob(os.path.join(args.idx_files, 'train*.idx')))
    val_files = sorted(glob.glob(os.path.join(args.tfrecords, 'val*.tfrec')))
    val_idx_files = sorted(glob.glob(os.path.join(args.idx_files, 'val*.idx')))
    # compute number of steps/batches per epoch
    nstep_per_epoch = args.num_train_samples // (args.batch_size*hvd.size())
    # compute number of steps/batches to iterate over entire validation set
    val_steps = args.num_val_samples // (args.batch_size*hvd.size())

    num_preprocessing_threads = 4
    train_preprocessor = DALIPreprocessor(train_files, train_idx_files, args.batch_size, num_preprocessing_threads, VECTOR_SIZE,
                                          dali_cpu=True, deterministic=False, training=False)
    val_preprocessor = DALIPreprocessor(val_files, val_idx_files, args.batch_size, num_preprocessing_threads, VECTOR_SIZE, dali_cpu=True,
                                        deterministic=False, training=False)

    train_input = train_preprocessor.get_device_dataset()
    val_input = val_preprocessor.get_device_dataset()

    train_labels_count = defaultdict(int)
    val_labels_count = defaultdict(int)

    for batch, (reads, labels) in enumerate(train_input.take(nstep_per_epoch), 1):
        if hvd.rank() == 0:
            print(reads.numpy()[0])
            print(labels.numpy())
        for l in labels.numpy():
            train_labels_count[l] += 1

    for batch, (reads, labels) in enumerate(val_input.take(val_steps), 1):
        if hvd.rank() == 0:
            print(reads.numpy()[0])
            print(labels.numpy())
        for l in labels.numpy():
            val_labels_count[l] += 1

    with open(os.path.join(args.output_dir, 'train_read_count'), 'w') as out_f:
        for k, v in train_labels_count.items():
            out_f.write(f'{k}\t{v}\n')

    with open(os.path.join(args.output_dir, 'val_read_count'), 'w') as out_f:
        for k, v in val_labels_count.items():
            out_f.write(f'{k}\t{v}\n')





if __name__ == "__main__":
    main()
