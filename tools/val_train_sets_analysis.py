import tensorflow as tf
import horovod.tensorflow as hvd
import tensorflow.keras as keras
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras import backend
from tensorflow.python.keras.mixed_precision import device_compatibility_check
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
import nvidia.dali.plugin.tf as dali_tf
import os
import sys
import json
import glob
import utils
import datetime
import numpy as np
import math
import io
import pandas as pd
from collections import defaultdict


# disable eager execution
#tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly())
# print which unit (CPU/GPU) is used for an operation
#tf.debugging.set_log_device_placement(True)

# enable XLA = XLA (Accelerated Linear Algebra) is a domain-specific compiler for linear algebra that can accelerate
# TensorFlow models with potentially no source code changes
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# Initialize Horovod
hvd.init()
# Pin GPU to be used to process local rank (one GPU per process)
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
    return reads, labels

class DALIPreprocessor(object):
    def __init__(self, filenames, idx_filenames, batch_size, num_threads, VECTOR_SIZE, dali_cpu=True,
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
            output_shapes=((batch_size, VECTOR_SIZE), (batch_size)),
            batch_size=batch_size, output_dtypes=(tf.int64, tf.int64), device_id=device_id)

    def get_device_dataset(self):
        return self.dalidataset

def parse_linclust(linclust_subset, linclust_dict):
    with open(linclust_subset, 'r') as f:
        for line in f:
            ref = line.rstrip().split('\t')[0]
            read = line.rstrip().split('\t')[1]
            linclust_dict[read] = ref

def main():
    input_dir = sys.argv[1]
    dataset_name = sys.argv[2]
    linclust_out = sys.argv[3]
    BATCH_SIZE = int(sys.argv[4])
    VECTOR_SIZE = 250 - 12 + 1
    num_train_samples = int(sys.argv[5])
    num_val_samples = int(sys.argv[6])
    # load training and validation tfrecords
    train_files = sorted(glob.glob(os.path.join(input_dir, f'tfrecords-{dataset_name}', f'training*')))
    train_idx_files = sorted(glob.glob(os.path.join(input_dir, f'tfrecords-{dataset_name}', 'idx_files', f'training*')))
    val_files = sorted(glob.glob(os.path.join(input_dir, f'tfrecords-{dataset_name}', f'validation*')))
    val_idx_files = sorted(glob.glob(os.path.join(input_dir, f'tfrecords-{dataset_name}', 'idx_files', f'validation*')))
    train_steps = num_train_samples // (BATCH_SIZE*hvd.size())
    val_steps = num_val_samples // (BATCH_SIZE*hvd.size())

    num_preprocessing_threads = 4

    train_preprocessor = DALIPreprocessor(train_files, train_idx_files, BATCH_SIZE, num_preprocessing_threads, VECTOR_SIZE,
                                          dali_cpu=True, deterministic=False, training=True)
    val_preprocessor = DALIPreprocessor(val_files, val_idx_files, BATCH_SIZE, num_preprocessing_threads, VECTOR_SIZE, dali_cpu=True,
                                        deterministic=False, training=False)

    train_input = train_preprocessor.get_device_dataset()
    val_input = val_preprocessor.get_device_dataset()


    # parse linclust output
    df = pd.read_csv(linclust_out, delimiter='\t')
    linclust_dict = df.to_dict()
    print(f'# reads: {len(df)}\t{len(linclust_dict)}')
    for key, value in linclust_dict.items():
        print(key, value)
        break

    # start = datetime.datetime.now()
    # for batch, (reads, labels) in enumerate(train_input.take(train_steps), 1):
    #     #print(reads, labels)
    #
    # end = datetime.datetime.now()
    #
    #
    #
    # total_time = end - start
    # hours, seconds = divmod(total_time.seconds, 3600)
    # minutes, seconds = divmod(seconds, 60)
    # # get true and predicted classes
    # f.write(f'Testing accuracy: {test_accuracy.result().numpy()*100}\n')
    # f.write(f'Testing loss: {test_loss.result().numpy()}\n')
    # f.write("Testing took %02d:%02d:%02d.%d\n" % (hours, minutes, seconds, total_time.microseconds))
    # print("\nTesting took %02d:%02d:%02d.%d\n" % (hours, minutes, seconds, total_time.microseconds))



if __name__ == "__main__":
    main()


# """ script to process linclust tsv output file and identify the number of identical reads between the training and validation sets """
# import sys
# import os
# import math
# from collections import defaultdict
# #from mpi4py import MPI
# import glob
# import multiprocess as mp
#
# def get_reads(dataset, fq_file, fq_files_loc):
#     with open(os.path.join(fq_files_loc, fq_file), 'r') as handle:
#         content = handle.readlines()
#         list_reads = [''.join(content[i:i+4]) for i in range(0, len(content), 4)]
#         for read in list_reads:
#             rec = read.split('\n')
#             dataset[rec[0]] = fq_file
#     print(mp.current_process(), fq_file, len(dataset))
#     return
#
# def parse_linclust(linclust_subset, training_set, validation_set, outfilename):
#     outfile = open(outfilename, 'w')
#     with open(linclust_subset, 'r') as f:
#         for line in f:
#             ref = line.rstrip().split('\t')[0]
#             read = line.rstrip().split('\t')[1]
#             if ref in training_set and read in validation_set:
#                 outfile.write(f'{ref}\t{training_set[ref]}\t{read}\t{validation_set[read]}\n')
#             elif ref in validation_set and read in training_set:
#                 outfile.write(f'{ref}\t{validation_set[ref]}\t{read}\t{training_set[read]}\n')
#             else:
#                 continue
#
# def parse_data(filename):
#     with open(os.path.join(filename), 'r') as f:
#         content = f.readlines()
#         content = [i.rstrip() for i in content]
#     print(len(content))
#     #chunk_size = math.ceil(len(content) // mp.cpu_count())
#     #chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
#     #print(filename, len(content), len(chunks), chunk_size, mp.cpu_count())
#     return content
#
# def main():
#     input_dir = sys.argv[1]
#     print(mp.cpu_count())
#     # parse data
#     train_fq_files_sets = parse_data(os.path.join(input_dir, 'training_data_cov_7x', 'list_fq_files'))
#     val_fq_files_sets = parse_data(os.path.join(input_dir, 'validation_data_cov_7x', 'list_fq_files'))
#     print(train_fq_files_sets)
#     print(val_fq_files_sets)
#     # get reads in training and validation sets
#     with mp.Manager() as manager:
#         training_set = manager.dict()
#         validation_set = manager.dict()
#         # fill out training dictionaries
#         processes = [mp.Process(target=get_reads, args=(training_set, fq_file, os.path.join(input_dir, 'training_data_cov_7x'))) for fq_file in train_fq_files_sets]
#         for p in processes:
#             p.start()
#         for p in processes:
#             p.join()
#         print(f'size of training set dictionary: {len(training_set)}')
#         # fill out validation dictionaries
#         processes = [mp.Process(target=get_reads, args=(validation_set, fq_file, os.path.join(input_dir, 'validation_data_cov_7x'))) for fq_file in val_fq_files_sets]
#         for p in processes:
#             p.start()
#         for p in processes:
#             p.join()
#         print(f'size of validation set dictionary: {len(validation_set)}')
#         # get number of necessary processes
#         num_processes = glob.glob(os.path.join(input_dir, f'linclust-subset-*'))
#         # parse linclust output data
#         processes = [mp.Process(target=parse_linclust, args=(os.path.join(input_dir, f'linclust-subset-{i}'), training_set, validation_set, os.path.join(input_dir, f'analysis-linclust-subset-{i}'))) for i in range(num_processes)]
#         for p in processes:
#             p.start()
#         for p in processes:
#             p.join()
#
#
#
#     # # create a communicator consisting of all the processors
#     # comm = MPI.COMM_WORLD
#     # # get the number of processors
#     # size = comm.Get_size()
#     # # get the rank of each processor
#     # rank = comm.Get_rank()
#     # print(comm, size, rank)
#     # if rank == 0:
#     #     # create dictionary for storing reads in training and validaiton sets
#     #     training_set = get_reads(os.path.join(input_dir, 'training_data_cov_7x'))
#     #     validation_set = get_reads(os.path.join(input_dir, 'validation_data_cov_7x'))
#     # else:
#     #     training_set = None
#     #     validation_set = None
#     #
#     # # broadcast dictionaries to other processes
#     # training_set = comm.bcast(training_set, root=0)
#     # validation_set = comm.bcast(validation_set, root=0)
#     #
#     # # load linclust output subset
#     # linclust_subset = open(os.path.join(input_dir, f'linclust-subset-{rank}'), 'r')
#     # print(f'{rank} - {linclust-subset}')
#     #
#     # # create output file
#     # outfile = open(f'analysis-linclust-subset-{rank}', 'w')
#     #
#     # # parse linclust output
#     # parse_linclust(linclust_subset, training_set, validation_set, outfile)
#
#
# if __name__ == "__main__":
#     main()
