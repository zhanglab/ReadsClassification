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


def run_testing(NUM_DEVICES, BATCH_SIZE, TESTING_STEPS, test_tfrecord, test_tfrecord_idx):
    # reset tensorflow graph
    tf.compat.v1.reset_default_graph()

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
                                                      "read_id": tfrec.VarLenFeature(tf.string)})

        def define_graph(self):
            inputs = self.input()
            reads = inputs["read"]
            read_id = inputs["read_id"]
            read_id = read_id.gpu()
            return (reads, read_id)


    # create an instance of strategy to perform synchronous training across multiple gpus
    strategy = tf.distribute.MirroredStrategy(devices=['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3'])

    with strategy.scope():
        # load model
        model = tf.keras.models.load_model(model)
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
        print(predictions.shape)
        end = datetime.datetime.now()
        total_time = end - start
        hours, seconds = divmod(total_time.seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        print("\nTook %02d:%02d:%02d.%d\n" % (hours, minutes, seconds, total_time.microseconds))
        f.write(f'runtime: {hours}:{minutes}:{seconds}.{total_time.microseconds}')
        f.close()

        test_read_ids = []
        def get_read_ids(inputs):
            global test_read_ids
            _, read_ids = inputs
            read_ids_arr = read_ids.numpy()
            test_read_ids += read_ids_arr.tolist()

        num_steps = 0
        for inputs in test_dataset:
            num_steps += 1
            strategy.run(get_read_ids, args=(inputs,))
            if num_steps == TESTING_STEPS:
                break

        # get predicted classes
        predicted_classes = []  # predicted classes as integers
        dict_conf_scores = {}  # confidence score
        for i in range(len(predictions)):
            pred_class = np.argmax(predictions[i])
            predicted_classes.append(pred_class)
            dict_conf_scores[test_read_ids[i]] = predictions[i][pred_class]

        return test_read_ids, predicted_classes, dict_conf_scores


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help="Path to input directory to store results")
    parser.add_argument('--model',  type=str, help="Path to model")
    parser.add_argument('--sample', type=str, help="Sample to test")
    parser.add_argument('--num_reads_tested', type=int, help="Number of reads in testing set")
    parser.add_argument('--threshold', type=float, help="Threshold for confidence score", default=0.5)

    model_num = int(args.model.split('/')[-1].split('-')[2]) + 1

    f = open(os.path.join(args.input_path, 'class_mapping.json'))
    args.class_mapping = json.load(f)

    NUM_CLASSES = len(class_mapping)
    NUM_DEVICES = 4  # number of GPUs
    BATCH_SIZE = 250  # batch size per GPU
    GLOBAL_BATCH_SIZE = NUM_DEVICES * BATCH_SIZE
    TESTING_SIZE = args.num_reads_tested
    TESTING_STEPS = math.ceil(TESTING_SIZE / GLOBAL_BATCH_SIZE)

    test_tfrecord = os.path.join(args.input_path, 'tfrecords', f'{args.sample}.tfrec')
    test_tfrecord_idx = os.path.join(args.input_path, 'tfrecords', f'idx_files/{args.sample}.tfrec.idx')

    # write down settings for training
    f = open(os.path.join(args.input_path, f'testing-info-model{model_num}'), 'w')
    f.write(
            f'model tested: {model}\n'
            f'sample tested: {sample}'
            f'batch size: {BATCH_SIZE}\n'
            f'global batch size: {GLOBAL_BATCH_SIZE}\n'
            f'reads in testing set: {TESTING_SIZE}\ntesting steps: {TESTING_STEPS}\n'
            f'number of devices: {NUM_DEVICES}\nnumber of classes: {NUM_CLASSES}\n'
    )

    # run testing
    test_read_ids, predicted_classes, dict_conf_scores = run_testing(NUM_DEVICES, BATCH_SIZE, TESTING_STEPS, test_tfrecord, test_tfrecord_idx)
    # create directory to store bins
    if not os.path.isdir(os.path.join(args.input_path, 'bins')):
        os.makedirs(os.path.join(args.input_path, 'bins'))
    # create bins
    create_bins(args, test_read_ids, predicted_classes, dict_conf_scores)


if __name__ == "__main__":
    main()