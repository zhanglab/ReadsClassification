import datetime
print(f'import tensorflow: {datetime.datetime.now()}')
import tensorflow as tf
# print(f'import nvidia dali: {datetime.datetime.now()}')
# from nvidia.dali.pipeline import pipeline_def
# import nvidia.dali.fn as fn
# import nvidia.dali.types as types
# import nvidia.dali.tfrecord as tfrec
# import nvidia.dali.plugin.tf as dali_tf
print(f'import model: {datetime.datetime.now()}')
from models import AlexNet
print(f'import remaining libraries: {datetime.datetime.now()}')
import os
import sys
import json
import glob
import numpy as np
import math
import gzip
from collections import defaultdict
import argparse
import multiprocessing as mp
import itertools

print(f'start code: {datetime.datetime.now()}')
print(f'# of cpu cores: {os.cpu_count()}')
# disable eager execution
#tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly())
# print which unit (CPU/GPU) is used for an operation
#tf.debugging.set_log_device_placement(True)

# enable XLA = XLA (Accelerated Linear Algebra) is a domain-specific compiler for linear algebra that can accelerate
# TensorFlow models with potentially no source code changes
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
print(f'end initialize code: {datetime.datetime.now()}')

# define the DALI pipeline
# @pipeline_def
# def get_dali_pipeline(tfrec_filenames, tfrec_idx_filenames):
#     inputs = fn.readers.tfrecord(path=tfrec_filenames,
#                                  index_path=tfrec_idx_filenames,
#                                  random_shuffle=False,
#                                  features={
#                                      "read": tfrec.VarLenFeature([], tfrec.int64, 0),
#                                      "read_id": tfrec.FixedLenFeature([1], tfrec.int64, -1)})
#     # retrieve reads and labels
#     reads = inputs["read"]
#     labels = inputs["read_id"]
#     return reads, labels
#
# class DALIPreprocessor(object):
#     def __init__(self, filenames, idx_filenames, batch_size, num_preprocessing_threads):
#         self.pipe = get_dali_pipeline(tfrec_filenames=filenames, tfrec_idx_filenames=idx_filenames, batch_size=batch_size, num_threads=num_preprocessing_threads, device_id=None, exec_pipelined=False)
#         self.dalidataset = dali_tf.DALIDataset(pipeline=self.pipe,
#             output_shapes=((batch_size, 239), (batch_size)),
#             batch_size=batch_size, output_dtypes=(tf.int64, tf.int64))
#
#     def get_device_dataset(self):
#         return self.dalidataset

@tf.function
def testing_step(reads, labels, model, loss=None, test_loss=None, test_accuracy=None):
    probs = model(reads, training=False)
    if test_loss != None and test_accuracy != None and loss != None:
        test_accuracy.update_state(labels, probs)
        loss_value = loss(labels, probs)
        test_loss.update_state(loss_value)
    pred_labels = tf.math.argmax(probs, axis=1)
    pred_probs = tf.reduce_max(probs, axis=1)
    return pred_labels, pred_probs
    # return probs

def read_tfrecord(proto_example):
    data_description = {
        'read': tf.io.VarLenFeature(tf.int64),
        'read_id': tf.io.VarLenFeature(tf.string)
    }
    # load one example
    parsed_example = tf.io.parse_single_example(serialized=proto_example, features=data_description)
    read = parsed_example['read']
    read = tf.sparse.to_dense(read)
    label = parsed_example['read_id']
    return read, label


def run_testing(args, results_dict, test_file):
    num_reads_classified = 0
    # get number of reads in test file
    with open(os.path.join(args.tfrecords, '-'.join([test_file, 'read_count'])), 'r') as f:
        num_reads = int(f.readline())
    num_reads_classified += num_reads
    # compute number of required steps to iterate over entire test file
    test_steps = math.ceil(num_reads/(args.batch_size))

    tfrec = os.path.join(args.tfrecords, '.'.join([test_file, 'tfrec']))
    idx_file = os.path.join(args.tfrecords, 'idx_files', '.'.join([test_file, 'tfrec.idx']))

    dataset = tf.data.TFRecordDataset([tfrec])
    dataset = dataset.map(map_func=read_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # num_preprocessing_threads = 1
    # test_preprocessor = DALIPreprocessor(tfrec, idx_file, args.batch_size, num_preprocessing_threads)
    # test_input = test_preprocessor.get_device_dataset()

    # create empty arrays to store the predicted and true values
    # all_predictions = tf.zeros([args.batch_size, NUM_CLASSES], dtype=tf.dtypes.float32, name=None)
    all_pred_sp = [tf.zeros([args.batch_size], dtype=tf.dtypes.float32, name=None)]
    all_prob_sp = [tf.zeros([args.batch_size], dtype=tf.dtypes.float32, name=None)]
    all_labels = [tf.zeros([args.batch_size], dtype=tf.dtypes.float32, name=None)]

    for batch, (reads, labels) in enumerate(dataset.take(test_steps), 1):
        if args.data_type == 'meta':
            batch_pred_sp, batch_prob_sp = testing_step(reads, labels, model)
            # batch_predictions = testing_step(reads, labels, model)
        elif args.data_type == 'test':
            batch_pred_sp, batch_prob_sp = testing_step(reads, labels, model, loss, test_loss, test_accuracy)
            # batch_predictions = testing_step(reads, labels, model, loss, test_loss, test_accuracy)

        if batch == 1:
            all_labels = [labels]
            all_pred_sp = [batch_pred_sp]
            all_prob_sp = [batch_prob_sp]
            # all_predictions = batch_predictions
        else:
            # all_predictions = tf.concat([all_predictions, batch_predictions], 0)
            all_pred_sp = tf.concat([all_pred_sp, [batch_pred_sp]], 1)
            all_prob_sp = tf.concat([all_prob_sp, [batch_prob_sp]], 1)
            all_labels = tf.concat([all_labels, [labels]], 1)

    # get list of true species, predicted species and predicted probabilities
    # all_predictions = all_predictions.numpy()
    # pred_species = [np.argmax(j) for j in all_predictions]
    # pred_probabilities = [np.amax(j) for j in all_predictions]
    all_pred_sp = all_pred_sp[0].numpy()
    all_prob_sp = all_prob_sp[0].numpy()
    all_labels = all_labels[0].numpy()

    # adjust the list of predicted species and read ids if necessary
    if len(all_pred_sp) > num_reads:
        num_extra_reads = (test_steps*args.batch_size) - num_reads
        # pred_species = pred_species[:-num_extra_reads]
        # pred_probabilities = pred_probabilities[:-num_extra_reads]
        all_pred_sp = all_pred_sp[:-num_extra_reads]
        all_prob_sp = all_prob_sp[:-num_extra_reads]
        all_labels = all_labels[:-num_extra_reads]

    # get dictionary mapping read ids to labels
    with open(os.path.join(args.tfrecords, '-'.join([test_file, 'read_ids.tsv'])), 'r') as f:
        content = f.readlines()
        dict_read_ids = {content[j].rstrip().split('\t')[1]: '@' + content[j].rstrip().split('\t')[0] for j in range(len(content))}

    for i in range(len(all_labels)):
        results_dict[all_labels[i]] = [args.class_mapping[str(all_pred_sp[i])], all_prob_sp[i]]



def main():
    start = datetime.datetime.now()
    print(f'start parsing command line arguments: {datetime.datetime.now()}')
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecords', type=str, help='path to tfrecords', required=True)
    parser.add_argument('--dali_idx', type=str, help='path to dali indexes files', required=True)
    parser.add_argument('--data_type', type=str, help='path to dali indexes files', required=True, choices=['test', 'meta'])
    # parser.add_argument('--fq_files', type=str, help='path to directory containing metagenomic data')
    parser.add_argument('--class_mapping', type=str, help='path to json file containing dictionary mapping taxa to labels', required=True)
    parser.add_argument('--output_dir', type=str, help='directory to store results', required=True)
    parser.add_argument('--epoch', type=int, help='epoch of checkpoint')
    parser.add_argument('--batch_size', type=int, help='batch size per gpu', default=512)
    parser.add_argument('--model', type=str, help='path to directory containing saved model')
    parser.add_argument('--ckpt', type=str, help='path to directory containing checkpoint file', required=('--epoch' in sys.argv))
    args = parser.parse_args()

    # define some training and model parameters
    VECTOR_SIZE = 250 - 12 + 1
    VOCAB_SIZE = 8390657
    EMBEDDING_SIZE = 60
    DROPOUT_RATE = 0.7

    # load class_mapping file mapping label IDs to species
    f = open(args.class_mapping)
    class_mapping = json.load(f)
    NUM_CLASSES = len(class_mapping)
    # create dtype policy
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

    # define metrics
    if args.data_type == 'test':
        loss = tf.losses.SparseCategoricalCrossentropy()
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    init_lr = 0.0001
    opt = tf.keras.optimizers.Adam(init_lr)
    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    print(f'create output directory: {datetime.datetime.now()}')
    # create output directories
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # load model
    if args.ckpt is not None:
        model = AlexNet(VECTOR_SIZE, EMBEDDING_SIZE, NUM_CLASSES, VOCAB_SIZE, DROPOUT_RATE)
        checkpoint = tf.train.Checkpoint(optimizer=opt, model=model)
        checkpoint.restore(os.path.join(args.ckpt, f'ckpts-{args.epoch}')).expect_partial()
    elif args.model is not None:
        model = tf.keras.models.load_model(args.model, 'model')
            # restore the checkpointed values to the model
    #        checkpoint = tf.train.Checkpoint(model)
    #        checkpoint.restore(tf.train.latest_checkpoint(os.path.join(input_dir, f'run-{run_num}', 'ckpts')))
    #        ckpt_path = os.path.join(input_dir, f'run-{run_num}', 'ckpts/ckpts')
    #        latest_ckpt = tf.train.latest_checkpoint(os.path.join(input_dir, f'run-{run_num}', 'ckpts'))
    #        print(f'latest ckpt: {latest_ckpt}')
    #        model.load_weights(os.path.join(input_dir, f'run-{run_num}', f'ckpts/ckpts-{epoch}'))

    print(f'load data: {datetime.datetime.now()}')
    # get list of testing tfrecords and number of reads per tfrecords
    test_files = [i.split('/')[-1].split('.')[0] for i in sorted(glob.glob(os.path.join(args.tfrecords, '*.tfrec')))]

    # split tfrecords between threads
    # test_files_per_gpu = math.ceil(len(test_files)/os.cpu_count())
    # threads_files = [test_files[i:i+test_files_per_gpu] for i in range(0, len(test_files), test_files_per_gpu)]

    manager = mp.Manager()
    results_dict = manager.dict()
    pool = mp.pool.ThreadPool(os.cpu_count())
    results = pool.starmap(run_testing, zip(itertools.repeat(args, len(test_files)), itertools.repeat(results_dict, len(test_files)), test_files))
    pool.close()
    pool.join()

    with open(os.path.join(args.output_dir, 'results.tsv'), 'w') as out_f:
        for key, value in results_dict.items():
            out_f.write(f'{key}\t{value[0]}\t{value[1]}\n')

    # with open(os.path.join(args.output_dir, f'{gpu_test_files[i].split("/")[-1].split(".")[0]}-out.tsv'), 'w') as out_f:
    #     for j in range(num_reads):
    #         # gpu_bins[str(pred_species[j])].append(all_read_ids[j])
    #         out_f.write(f'{dict_read_ids[str(all_labels[j])]}\t{all_pred_sp[j]}\t{all_prob_sp[j]}\n')
            # out_f.write(f'{dict_read_ids[str(all_labels[j])]}\t{pred_species[j]}\t{pred_probabilities[j]}\n')


    # num_reads_classified = 0
    # for i in range(len(gpu_test_files)):
    #     # get number of reads in test file
    #     with open(os.path.join(args.tfrecords, gpu_num_reads_files[i]), 'r') as f:
    #         num_reads = int(f.readline())
    #     num_reads_classified += num_reads
    #     # compute number of required steps to iterate over entire test file
    #     test_steps = math.ceil(num_reads/(args.batch_size))
    #
    #     num_preprocessing_threads = 4
    #     test_preprocessor = DALIPreprocessor(gpu_test_files[i], gpu_test_idx_files[i], args.batch_size, num_preprocessing_threads, dali_cpu=True, deterministic=False, training=False)
    #
    #     test_input = test_preprocessor.get_device_dataset()
    #
    #     # create empty arrays to store the predicted and true values
    #     # all_predictions = tf.zeros([args.batch_size, NUM_CLASSES], dtype=tf.dtypes.float32, name=None)
    #     all_pred_sp = [tf.zeros([args.batch_size], dtype=tf.dtypes.float32, name=None)]
    #     all_prob_sp = [tf.zeros([args.batch_size], dtype=tf.dtypes.float32, name=None)]
    #     all_labels = [tf.zeros([args.batch_size], dtype=tf.dtypes.float32, name=None)]
    #
    #     for batch, (reads, labels) in enumerate(test_input.take(test_steps), 1):
    #         if args.data_type == 'meta':
    #             batch_pred_sp, batch_prob_sp = testing_step(reads, labels, model)
    #             # batch_predictions = testing_step(reads, labels, model)
    #         elif args.data_type == 'test':
    #             batch_pred_sp, batch_prob_sp = testing_step(reads, labels, model, loss, test_loss, test_accuracy)
    #             # batch_predictions = testing_step(reads, labels, model, loss, test_loss, test_accuracy)
    #
    #         if batch == 1:
    #             all_labels = [labels]
    #             all_pred_sp = [batch_pred_sp]
    #             all_prob_sp = [batch_prob_sp]
    #             # all_predictions = batch_predictions
    #         else:
    #             # all_predictions = tf.concat([all_predictions, batch_predictions], 0)
    #             all_pred_sp = tf.concat([all_pred_sp, [batch_pred_sp]], 1)
    #             all_prob_sp = tf.concat([all_prob_sp, [batch_prob_sp]], 1)
    #             all_labels = tf.concat([all_labels, [labels]], 1)
    #
    #
    #
    #     # get list of true species, predicted species and predicted probabilities
    #     # all_predictions = all_predictions.numpy()
    #     # pred_species = [np.argmax(j) for j in all_predictions]
    #     # pred_probabilities = [np.amax(j) for j in all_predictions]
    #     all_pred_sp = all_pred_sp[0].numpy()
    #     all_prob_sp = all_prob_sp[0].numpy()
    #     all_labels = all_labels[0].numpy()
    #
    #     # adjust the list of predicted species and read ids if necessary
    #     if len(all_pred_sp) > num_reads:
    #         num_extra_reads = (test_steps*args.batch_size) - num_reads
    #         # pred_species = pred_species[:-num_extra_reads]
    #         # pred_probabilities = pred_probabilities[:-num_extra_reads]
    #         all_pred_sp = all_pred_sp[:-num_extra_reads]
    #         all_prob_sp = all_prob_sp[:-num_extra_reads]
    #         all_labels = all_labels[:-num_extra_reads]

        # fill out dictionary of bins and create summary file of predicted probabilities
        # gpu_bins = {label: [] for label in class_mapping.keys()} # key = species predicted, value = list of read ids

        # # get dictionary mapping read ids to labels
        # with open(os.path.join(args.tfrecords, gpu_read_ids_files[i]), 'r') as f:
        #     content = f.readlines()
        #     dict_read_ids = {content[j].rstrip().split('\t')[1]: '@' + content[j].rstrip().split('\t')[0] for j in range(len(content))}
        #
        # with open(os.path.join(args.output_dir, f'{gpu_test_files[i].split("/")[-1].split(".")[0]}-out.tsv'), 'w') as out_f:
        #     for j in range(num_reads):
        #         # gpu_bins[str(pred_species[j])].append(all_read_ids[j])
        #         out_f.write(f'{dict_read_ids[str(all_labels[j])]}\t{all_pred_sp[j]}\t{all_prob_sp[j]}\n')
                # out_f.write(f'{dict_read_ids[str(all_labels[j])]}\t{pred_species[j]}\t{pred_probabilities[j]}\n')



        # get reads
        # with gzip.open(os.path.join(args.fq_files, f'{gpu_test_files[i].split("/")[-1].split(".")[0]}.fastq.gz'), 'rt') as f:
        #     content = f.readlines()
        #     records = [''.join(content[j:j+4]) for j in range(0, len(content), 4)]
        #     reads = {records[j].split('\n')[0]: records[j] for j in range(len(records))}

        # report species abundance and create bins
        # with open(os.path.join(args.output_dir, f'{gpu_test_files[i].split("/")[-1].split(".")[0]}-results.tsv'), 'w') as out_f:
            # for key, value in gpu_bins.items():
                # out_f.write(f'{key}\t{len(value)}\n')
                # if len(value) > 0:
                #     # create fastq files from bins
                #     with open(os.path.join(args.output_dir, f'{gpu_test_files[i].split("/")[-1].split(".")[0]}-bin-{key}.fq'), 'w') as out_fq:
                #         list_reads_in_bin = [reads[dict_read_ids[str(j)]] for j in value]
                #         out_fq.write(''.join(list_reads_in_bin))

    print(f'end testing: {datetime.datetime.now()}')

    end = datetime.datetime.now()
    total_time = end - start
    hours, seconds = divmod(total_time.seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    with open(os.path.join(args.output_dir, 'testing-summary.tsv'), 'w') as outfile:
        outfile.write(f'batch size per gpu: {args.batch_size}\nnumber of threads: {os.cpu_count()}\nnumber of tfrecord files: {len(test_files)}\n')
        if args.ckpt:
            outfile.write(f'checkpoint saved at epoch: {args.epoch}')
        else:
            outfile.write(f'model saved at last epoch')
        outfile.write(f'\nrun time: {hours}:{minutes}:{seconds}:{total_time.microseconds}')


if __name__ == "__main__":
    main()
