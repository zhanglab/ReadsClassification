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

from summarize import create_barplot_testing, ROCcurve, metrics_report

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

tf.compat.v1.reset_default_graph()

input_path = str(sys.argv[1])
model_path = str(sys.argv[2])
model_name = str(model_path.split('/')[-2])
epoch = int(model_path.split('/')[-1].split('-')[1]) + 1
tfrecords_path = str(sys.argv[3])
path_to_data = str(sys.argv[4])
num_gpus = int(sys.argv[5])

output_path = os.path.join(input_path, model_name, f'testing-epoch{epoch}')
if not os.path.isdir(output_path):
    os.makedirs(output_path)

f = open(os.path.join(input_path, 'class_mapping.json'))
class_mapping = json.load(f)

# load colors associated with each species
colors = []
if os.path.isfile(os.path.join(input_path, 'colors')):
    print('file exists')
    with open(os.path.join(input_path, 'colors'), 'r') as f:
        for line in f:
            line = line.rstrip().split('\t')
            colors.append(line[1])
else:
    print('not exists')
    with open(os.path.join(input_path, 'colors'), 'w') as f:
        for i in range(len(class_mapping)):
            color = '#%06X' % randint(0, 0xFFFFFF)
            colors.append(color)
            f.write(f'{i}\t{color}\n')
print(colors)


K_VALUE = 12
READ_LENGTH = 250
NUM_CLASSES = len(class_mapping)
NUM_DEVICES = num_gpus  # number of GPUs
BATCH_SIZE = 250  # batch size per GPU
GLOBAL_BATCH_SIZE = NUM_DEVICES * BATCH_SIZE
VOCAB_SIZE = 8390658
EPOCHS = 50
EMBEDDING_SIZE = 60
VECTOR_SIZE = READ_LENGTH - K_VALUE + 1
TESTING_SIZE = 10725392
TESTING_STEPS = math.ceil(TESTING_SIZE / GLOBAL_BATCH_SIZE)

test_tfrecord = os.path.join(tfrecords_path, f'testing_data.tfrec')
test_tfrecord_idx = os.path.join(tfrecords_path, f'idx_files/testing_data.tfrec.idx')

# write down settings for training
f = open(os.path.join(output_path, f'testing-info-epoch{epoch}'), 'w')
f.write(f'batch size: {BATCH_SIZE}\n'
        f'global batch size: {GLOBAL_BATCH_SIZE}\n'
        f'reads in testing set: {TESTING_SIZE}\ntesting steps: {TESTING_STEPS}\n'
        f'read length: {READ_LENGTH}\nk value: {K_VALUE}\nnumber of devices: {NUM_DEVICES}\nnumber of classes: {NUM_CLASSES}\n'
        f'embedding size: {EMBEDDING_SIZE}\nnumber of kmers: {VOCAB_SIZE}\ninput size: {VECTOR_SIZE}\n')

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


dict_gpus = {'1': '/gpu:0', '2': '/gpu:0, /gpu:1', '3': '/gpu:0, /gpu:1, /gpu:2', '4': '/gpu:0, /gpu:1, /gpu:2, /gpu:3'}
# create an instance of strategy to perform synchronous training across multiple gpus
strategy = tf.distribute.MirroredStrategy(devices=[dict_gpus[str(NUM_DEVICES)]])

with strategy.scope():
    # load model
    model = tf.keras.models.load_model(model_path)
    print('model loaded')

    def test_dataset_fn(input_context):
        with tf.device("/gpu:{}".format(input_context.input_pipeline_id)):
            device_id = input_context.input_pipeline_id
            return dali_tf.DALIDataset(fail_on_device_mismatch=False,
                                       pipeline=TFRecordPipeline(BATCH_SIZE, test_tfrecord, test_tfrecord_idx,
                                                                 device_id=device_id, shard_id=device_id,
                                                                 num_shards=NUM_DEVICES),
                                       batch_size=BATCH_SIZE, output_shapes=shapes, output_dtypes=dtypes,
                                       device_id=device_id)


    input_options = tf.distribute.InputOptions(
        experimental_place_dataset_on_device=True,
        experimental_prefetch_to_device=False,
        experimental_replication_mode=tf.distribute.InputReplicationMode.PER_REPLICA)

    test_dataset = strategy.distribute_datasets_from_function(test_dataset_fn, input_options)
    print("Make predictions on testing data")
    start = datetime.datetime.now()

    #    model.evaluate(test_dataset, batch_size=BATCH_SIZE, steps=TESTING_STEPS)
    predictions = model.predict(test_dataset, batch_size=BATCH_SIZE, steps=TESTING_STEPS)
    predictions = predictions[:TESTING_SIZE]
    end = datetime.datetime.now()
    total_time = end - start
    hours, seconds = divmod(total_time.seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    print("\nTook %02d:%02d:%02d.%d\n" % (hours, minutes, seconds, total_time.microseconds))
    f.write(f'runtime: {hours}:{minutes}:{seconds}.{total_time.microseconds}')
    f.close()

    # get predicted classes
    predicted_classes = []  # predicted classes as integeres
    for i in range(len(predictions)):
        pred_class = np.argmax(predictions[i])
        predicted_classes.append(pred_class)

    test_true_classes_int = []

    def get_labels(inputs):
        global test_true_classes_int
        global test_true_classes_vec
        _, labels = inputs
        t_classes = labels.numpy()
        test_true_classes_int += t_classes.tolist()

    num_steps = 0
    for inputs in test_dataset:
        num_steps += 1
        strategy.run(get_labels, args=(inputs, ))
        if num_steps == TESTING_STEPS:
            break
    
    # convert integers to one hot vectors
    test_true_classes_int = test_true_classes_int[:TESTING_SIZE]
    test_true_classes_vec = to_categorical(test_true_classes_int, num_classes=NUM_CLASSES)
    print(f'test: {len(test_true_classes_int)} - {len(test_true_classes_vec)} - {len(predictions)} - {len(predicted_classes)}')

    test_dict_classes = defaultdict(int)
    for j in range(len(test_true_classes_int)):
        test_dict_classes[test_true_classes_int[j]] += 1

    create_barplot_testing(test_dict_classes, os.path.join(output_path, f'data-barplots-testing'), class_mapping)
 
    list_labels = [class_mapping[str(i)] for i in range(len(class_mapping))]
    # get precision and recall for each class
    metrics_report(test_true_classes_int, predicted_classes, list_labels, output_path, class_mapping, epoch, path_to_data)
    # create ROC curves
    ROCcurve(test_dict_classes, test_true_classes_vec, predictions, class_mapping, output_path, epoch, colors)
