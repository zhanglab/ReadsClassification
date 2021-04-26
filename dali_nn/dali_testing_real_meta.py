import nvidia.dali as dali
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.plugin.tf as dali_tf
import nvidia.dali.tfrecord as tfrec

import tensorflow as tf
from tensorflow import keras
from keras.utils.np_utils import to_categorical
from random import randint
import numpy as np
import json
from collections import defaultdict
import os
import sys
import io
import math
import datetime
import argparse

from binning import create_bins


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

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, help="Path to directory to store results")
parser.add_argument('--model',  type=str, help="Path to model")
parser.add_argument('--sample', type=str, help="Sample to test")
parser.add_argument('--num_reads_tested', type=int, help="Number of reads in testing set")
parser.add_argument('--threshold', type=float, help="Threshold for confidence score", default=0.5)
parser.add_argument('--path_to_sample', type=str, help="Path to tfrecords of metagenomic data")
parser.add_argument('--k_value', type=int, help="k value", default=12)
parser.add_argument('--read_length', type=int, help="read length", default=250)
parser.add_argument('--read_id_ms', type=int, help="read ids max size")
parser.add_argument('--gpus', type=int, help="number of gpus", default=1)
args = parser.parse_args()

args.model_num = int(args.model.split('/')[-2][-1])
args.epoch_num = int(args.model.split('/')[-1].split('-')[1]) + 1
args.output_path = os.path.join(args.input_path, f'model{args.model_num}')

f = open(os.path.join(args.input_path, 'class_mapping.json'))
args.class_mapping = json.load(f)

NUM_CLASSES = len(args.class_mapping)
NUM_DEVICES = args.gpus  # number of GPUs
BATCH_SIZE = 500  # batch size per GPU
GLOBAL_BATCH_SIZE = NUM_DEVICES * BATCH_SIZE
TESTING_SIZE = args.num_reads_tested
TESTING_STEPS = math.ceil(TESTING_SIZE / GLOBAL_BATCH_SIZE)
VECTOR_SIZE = args.read_length - args.k_value + 1

fw_tfrecord = os.path.join(args.path_to_sample, 'tfrecords', f'{args.sample}_R1.tfrec')
fw_tfrecord_idx = os.path.join(args.path_to_sample, 'tfrecords', f'idx_files/{args.sample}_R1.tfrec.idx')
rv_tfrecord = os.path.join(args.path_to_sample, 'tfrecords', f'{args.sample}_R2.tfrec')
rv_tfrecord_idx = os.path.join(args.path_to_sample, 'tfrecords', f'idx_files/{args.sample}_R2.tfrec.idx')

# write down settings for training
f = open(os.path.join(args.output_path, f'sample-{args.sample}-classification-model{args.model_num}-epoch{args.epoch_num}'), 'w')
f.write(
        f'model tested: model{args.model_num}\n'
        f'epoch number: {args.epoch_num}\n'
        f'sample tested: {args.sample}\n'
        f'batch size: {BATCH_SIZE}\n'
        f'global batch size: {GLOBAL_BATCH_SIZE}\n'
        f'reads in testing set: {TESTING_SIZE}\ntesting steps: {TESTING_STEPS}\n'
        f'number of devices: {NUM_DEVICES}\nnumber of classes: {NUM_CLASSES}\n'
)


# reset tensorflow graph
tf.compat.v1.reset_default_graph()

# define shapes and types of data and labels
shapes = (
        (BATCH_SIZE, VECTOR_SIZE),
        (BATCH_SIZE, args.read_id_ms+1))
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


class TFRecordPipeline(Pipeline):
    def __init__(self, batch_size, tfrecord, tfrecord_idx, device_id=0, shard_id=0, num_shards=1, num_threads=4,
                     seed=0):
        super(TFRecordPipeline, self).__init__(batch_size, num_threads, device_id, seed)
        self.input = ops.TFRecordReader(path=tfrecord, random_shuffle=False, shard_id=shard_id, num_shards=num_shards,
                                            index_path=tfrecord_idx,
                                            features={"read": tfrec.VarLenFeature([], tfrec.int64, 0),
                                                      "read_id": tfrec.VarLenFeature([], tfrec.int64, 64)})

    def define_graph(self):
        inputs = self.input()
        reads = inputs["read"]
        read_id = inputs["read_id"]
        read_id = read_id.gpu()
        return (reads, read_id)


# create an instance of strategy to perform synchronous training across multiple gpus
#strategy = tf.distribute.MirroredStrategy(devices=['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3'])
strategy = tf.distribute.MirroredStrategy(devices=['/gpu:0'])
with strategy.scope():
    # load model
    model = tf.keras.models.load_model(args.model)
    print('model loaded')

    def fw_dataset_fn(input_context):
        with tf.device("/gpu:{}".format(input_context.input_pipeline_id)):
            device_id = input_context.input_pipeline_id
            return dali_tf.DALIDataset(fail_on_device_mismatch=False,
                                           pipeline=TFRecordPipeline(BATCH_SIZE, fw_tfrecord, fw_tfrecord_idx,
                                                                     device_id=device_id, shard_id=device_id,
                                                                     num_shards=NUM_DEVICES),
                                           batch_size=BATCH_SIZE, output_shapes=shapes, output_dtypes=dtypes,
                                           device_id=device_id)


    def rv_dataset_fn(input_context):
        with tf.device("/gpu:{}".format(input_context.input_pipeline_id)):
            device_id = input_context.input_pipeline_id
            return dali_tf.DALIDataset(fail_on_device_mismatch=False,
                                           pipeline=TFRecordPipeline(BATCH_SIZE, rv_tfrecord, rv_tfrecord_idx,
                                                                     device_id=device_id, shard_id=device_id,
                                                                     num_shards=NUM_DEVICES),
                                           batch_size=BATCH_SIZE, output_shapes=shapes, output_dtypes=dtypes,
                                           device_id=device_id)

    input_options = tf.distribute.InputOptions(
        experimental_place_dataset_on_device=True,
        experimental_prefetch_to_device=False,
        experimental_replication_mode=tf.distribute.InputReplicationMode.PER_REPLICA)

    fw_dataset = strategy.distribute_datasets_from_function(fw_dataset_fn, input_options)
    rv_dataset = strategy.distribute_datasets_from_function(rv_dataset_fn, input_options)
    print("Make predictions on testing data")
    start = datetime.datetime.now()
    #    model.evaluate(test_dataset, batch_size=BATCH_SIZE, steps=TESTING_STEPS)
    fw_predictions = model.predict(fw_dataset, batch_size=BATCH_SIZE, steps=TESTING_STEPS)
    rv_predictions = model.predict(rv_dataset, batch_size=BATCH_SIZE, steps=TESTING_STEPS)
    print(fw_predictions.shape, rv_predictions.shape)
    end = datetime.datetime.now()
    total_time = end - start
    hours, seconds = divmod(total_time.seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    print("\nTook %02d:%02d:%02d.%d\n" % (hours, minutes, seconds, total_time.microseconds))
    f.write(f'runtime: {hours}:{minutes}:{seconds}.{total_time.microseconds}')
    f.close()

    fw_read_ids = []
    def get_fw_read_ids(inputs):
        global fw_read_ids
        _, read_ids = inputs
        read_ids_arr = read_ids.numpy()
        read_ids_list = read_ids_arr.tolist()
        read_ids_str = []
        for i in range(len(read_ids_list)):
            r = read_ids_list[i]
            rl = r[args.read_id_ms]
            new_r = ''.join([chr(j) for j in r[0:rl]])
            read_ids_str.append(new_r)
        fw_read_ids += read_ids_str
    
    num_steps = 0
    for inputs in fw_dataset:
        num_steps += 1
        strategy.run(get_fw_read_ids, args=(inputs,))
        if num_steps == TESTING_STEPS:
            break
    print(f'fw num steps: {num_steps} - {TESTING_STEPS}')
    rv_read_ids = []
    def get_rv_read_ids(inputs):
        global rv_read_ids
        _, read_ids = inputs
        read_ids_arr = read_ids.numpy()
        read_ids_list = read_ids_arr.tolist()
        read_ids_str = []
        for r in read_ids_list:
            rl = r[args.read_id_ms]
            new_r = ''.join([chr(j) for j in r[0:rl]])
            read_ids_str.append(new_r)
        rv_read_ids += read_ids_str

    num_steps = 0
    for inputs in rv_dataset:
        num_steps += 1
        strategy.run(get_rv_read_ids, args=(inputs,))
        if num_steps == TESTING_STEPS:
            break
    print(f'rv num steps: {num_steps} - {TESTING_STEPS}')
    print(f'first fw reads in 2 consecutive rounds: {fw_read_ids[0]} - {fw_read_ids[args.num_reads_tested]}')
    print(f'first rv reads in 2 consecutive rounds: {rv_read_ids[0]} - {rv_read_ids[args.num_reads_tested]}')
    # adjust list of read ids
    fw_read_ids = fw_read_ids[0:args.num_reads_tested]
    rv_read_ids = rv_read_ids[0:args.num_reads_tested]
    # get predicted classes
    fw_predicted_classes = []  # predicted classes as integers
    fw_dict_conf_scores = {}  # confidence score
    for i in range(args.num_reads_tested):
        pred_class = np.argmax(fw_predictions[i])
        fw_predicted_classes.append(pred_class)
        fw_dict_conf_scores[fw_read_ids[i]] = float(fw_predictions[i][pred_class])
        print(fw_read_ids[i], pred_class, fw_predictions[i][pred_class])
        
    print(f'size set/list fw_read_ids: {len(set(fw_read_ids))}, {len(fw_read_ids)}')
    rv_predicted_classes = []  # predicted classes as integers
    rv_dict_conf_scores = {}  # confidence score
    for i in range(args.num_reads_tested):
        pred_class = np.argmax(rv_predictions[i])
        rv_predicted_classes.append(pred_class)
        rv_dict_conf_scores[rv_read_ids[i]] = float(rv_predictions[i][pred_class])
        print(rv_read_ids[i], pred_class, rv_predictions[i][pred_class])

    # create directory to store bins
    if not os.path.isdir(os.path.join(args.output_path, f'sample-{args.sample}-bins')):
        os.makedirs(os.path.join(args.output_path, f'sample-{args.sample}-bins'))
    # create bins
    create_bins(args, fw_read_ids, rv_read_ids, fw_predicted_classes, rv_predicted_classes, fw_dict_conf_scores, rv_dict_conf_scores)


