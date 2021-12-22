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
from collections import defaultdict
from models import AlexNet, VGG16, VDCNN
from resnet50 import ResNet50
from summarize import *

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
#for gpu in gpus:
#    tf.config.experimental.set_memory_growth(gpu, True)
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
    def __init__(self, filenames, idx_filenames, batch_size, num_threads, dali_cpu=True,
               deterministic=False, training=False):
        # each device receives the same list of tfrecords files
        device_id = hvd.local_rank()
        shard_id = hvd.rank()
        num_gpus = hvd.size()
        #print(f'{hvd.rank()}\t{filenames}\t{idx_filenames}')
        self.pipe = get_dali_pipeline(tfrec_filenames=filenames, tfrec_idx_filenames=idx_filenames, batch_size=batch_size,
                                      num_threads=num_threads, device_id=device_id, shard_id=shard_id, num_gpus=num_gpus,
                                      dali_cpu=dali_cpu, training=training, seed=7 * (1 + hvd.rank()) if deterministic else None)
        self.daliop = dali_tf.DALIIterator()

        self.batch_size = batch_size
        self.device_id = device_id

        self.dalidataset = dali_tf.DALIDataset(fail_on_device_mismatch=False, pipeline=self.pipe,
            output_shapes=((batch_size, 239), (batch_size)),
            batch_size=batch_size, output_dtypes=(tf.int64, tf.int64), device_id=device_id)

    def get_device_dataset(self):
        return self.dalidataset

@tf.function
def testing_step(reads, labels, loss, test_loss, test_accuracy, model):
    probs = model(reads, training=False)
    test_accuracy.update_state(labels, probs)
    loss_value = loss(labels, probs)
    test_loss.update_state(loss_value)

    return probs

def main():
    input_dir = sys.argv[1]
    run_num = sys.argv[2]
    dataset_name = sys.argv[3]
    set_type = sys.argv[4]
    path_to_tfrecords = sys.argv[5]
    # define some training and model parameters
    VECTOR_SIZE = 250 - 12 + 1
    VOCAB_SIZE = 8390657
    EMBEDDING_SIZE = 60
    DROPOUT_RATE = float(sys.argv[6])
    BATCH_SIZE = int(sys.argv[7])
    num_reads = int(sys.argv[8])
    num_reads_last = int(sys.argv[9])
    EPOCH = int(sys.argv[10])
    # load class_mapping file mapping label IDs to species
    f = open(os.path.join(input_dir, 'class_mapping.json'))
    class_mapping = json.load(f)
    NUM_CLASSES = len(class_mapping)
    # create dtype policy
    policy = keras.mixed_precision.Policy('mixed_float16')
    keras.mixed_precision.set_global_policy(policy)

    # define metrics
    loss = tf.losses.SparseCategoricalCrossentropy()
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    init_lr = 0.0001
    opt = tf.keras.optimizers.Adam(init_lr)
    opt = keras.mixed_precision.LossScaleOptimizer(opt)
 
    # load model
    if EPOCH is not None:
        model = AlexNet(input_dir, VECTOR_SIZE, EMBEDDING_SIZE, NUM_CLASSES, VOCAB_SIZE, DROPOUT_RATE, run_num)
        checkpoint = tf.train.Checkpoint(optimizer=opt, model=model)
        checkpoint.restore(os.path.join(input_dir, f'run-{run_num}', f'ckpts-{EPOCH}')).expect_partial()
    else:
        model = tf.keras.models.load_model(os.path.join(input_dir, f'run-{run_num}', 'ckpts/model'))

    # create empty confusion matrix with rows = true classes and columns = predicted classes
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES))
    num_correct_pred = 0
    num_incorrect_pred = 0 

    # create output directory
    output_dir = os.path.join(input_dir, f'run-{run_num}', f'testing-{set_type}-set', f'all-genomes-epoch-{EPOCH}', f'results-gpu-{hvd.rank()}')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    start = datetime.datetime.now()

    # load testing tfrecords
    test_files = sorted(glob.glob(os.path.join(path_to_tfrecords, f'tfrecords-{hvd.rank()}', '*.tfrec')))
    test_idx_files = sorted(glob.glob(os.path.join(path_to_tfrecords, f'tfrecords-{hvd.rank()}', 'idx_files', '*.tfrec.idx')))
    for i in range(len(test_files)):
        if i == len(test_files) - 1:
            num_reads = num_reads_last
        test_steps = math.ceil(num_reads/(BATCH_SIZE))
        num_preprocessing_threads = 4
        test_preprocessor = DALIPreprocessor(test_files[i], test_idx_files[i], BATCH_SIZE, num_preprocessing_threads, dali_cpu=True,
                                        deterministic=False, training=False)
        test_input = test_preprocessor.get_device_dataset()

        all_predictions = tf.zeros([BATCH_SIZE, NUM_CLASSES], dtype=tf.dtypes.float32, name=None)
        all_labels = [tf.zeros([BATCH_SIZE], dtype=tf.dtypes.float32, name=None)]

        for batch, (reads, labels) in enumerate(test_input.take(test_steps), 1):
            batch_predictions = testing_step(reads, labels, loss, test_loss, test_accuracy, model)
            if batch == 1:
                all_labels = [labels]
                all_predictions = batch_predictions
            elif batch > 1:
                all_predictions = tf.concat([all_predictions, batch_predictions], 0)
                all_labels = tf.concat([all_labels, [labels]], 1)


        # get list of true and predicted species
        true_species = all_labels[0].numpy().tolist()
        pred_species = [np.argmax(j) for j in all_predictions.numpy()]
        pred_prob = [np.amax(j) for j in all_predictions.numpy()]

        # adjust the list of predicted and true species if necessary
        print(f'{hvd.rank()}\tsize of pred_species: {len(pred_species)}\nsize of true_species: {len(true_species)}')
        if len(pred_species) > num_reads:
            num_extra_reads = (test_steps*BATCH_SIZE) - num_reads
            pred_species = pred_species[:-num_extra_reads]
            true_species = true_species[:-num_extra_reads]

        f_prob = open(os.path.join(output_dir, f'probabilities-gpu-{hvd.rank()}.tsv'), 'a')
        # fill out confusion matrix and output probabilities
        for i in range(len(true_species)):
            cm[true_species[i], pred_species[i]] += 1
            if true_species[i] == pred_species[i]:
                num_correct_pred += 1
                f_prob.write(f'correct\t{true_species[i]}\t{pred_species[i]}\t{pred_prob[i]}\n')
            else:
                num_incorrect_pred += 1
                f_prob.write(f'incorrect\t{true_species[i]}\t{pred_species[i]}\t{pred_prob[i]}\n')
        f_prob.close()

        # write results to file
        with open(os.path.join(output_dir, f'confusion-matrix-gpu-{hvd.rank()}.tsv'), 'w') as cm_f:
            for i in range(NUM_CLASSES):
                cm_f.write(f'\t{class_mapping[str(i)]}')
            cm_f.write('\n')
            for i in range(NUM_CLASSES):
                cm_f.write(f'{class_mapping[str(i)]}')
                for j in range(NUM_CLASSES):
                    cm_f.write(f'\t{cm[i,j]}')
                cm_f.write('\n')

    
        accuracy = round(float(num_correct_pred)/(num_correct_pred + num_incorrect_pred), 5)

    end = datetime.datetime.now()
    total_time = end - start
    hours, seconds = divmod(total_time.seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    with open(os.path.join(output_dir, f'testing-summary-gpu-{hvd.rank()}.tsv'), 'w') as outfile:
        outfile.write(f'run: {run_num}\ntfrecords: {dataset_name}\ntesting set: {set_type}\nnumber of classes: {NUM_CLASSES}\nvector size: {VECTOR_SIZE}\nvocabulary size: {VOCAB_SIZE}\nembedding size: {EMBEDDING_SIZE}\ndropout rate: {DROPOUT_RATE}\nbatch size per gpu: {BATCH_SIZE}\nnumber of gpus: 1\nGPU: {hvd.rank()}\nnumber of tfrecord files: {len(test_files)}\nnumber of reads per tfrecord file: {num_reads}\nnumber of reads in last tfrecord file: {num_reads_last}\nnumber of reads tested: {num_correct_pred + num_incorrect_pred}')
        outfile.write(f'\n{test_accuracy.result().numpy()*100}\n{accuracy}\n{test_loss.result().numpy()}\n{hours}:{minutes}:{seconds}:{total_time.microseconds}')


if __name__ == "__main__":
    main()
