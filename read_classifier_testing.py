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
import datetime
import numpy as np
import math
import io
from collections import defaultdict
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
    def __init__(self, filenames, idx_filenames, batch_size, num_threads, dali_cpu=True,
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
    test_path = sys.argv[2]
    run_num = sys.argv[3]
    set_type = sys.argv[4]
    # define some training and model parameters
    VECTOR_SIZE = 250 - 12 + 1
    VOCAB_SIZE = 8390657
    EMBEDDING_SIZE = 60
    DROPOUT_RATE = float(sys.argv[5])
    BATCH_SIZE = int(sys.argv[6])
    num_reads = int(sys.argv[7])
    epoch = int(sys.argv[8])
    test_steps = math.ceil(num_reads/(BATCH_SIZE*hvd.size()))
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

    if hvd.rank() == 0:
        # create output directory
        output_dir = os.path.join(input_dir, f'run-{run_num}', f'testing-{set_type}-set')
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        # restore the checkpointed values to the model
#        checkpoint = tf.train.Checkpoint(model)
#        checkpoint.restore(tf.train.latest_checkpoint(os.path.join(input_dir, f'run-{run_num}', 'ckpts')))
#        ckpt_path = os.path.join(input_dir, f'run-{run_num}', 'ckpts/ckpts')
#        latest_ckpt = tf.train.latest_checkpoint(os.path.join(input_dir, f'run-{run_num}', 'ckpts'))
#        print(f'latest ckpt: {latest_ckpt}')
#        model.load_weights(os.path.join(input_dir, f'run-{run_num}', f'ckpts/ckpts-{epoch}'))
        if epoch is not None:
            model = AlexNet(input_dir, VECTOR_SIZE, EMBEDDING_SIZE, NUM_CLASSES, VOCAB_SIZE, DROPOUT_RATE, run_num)
            checkpoint = tf.train.Checkpoint(optimizer=opt, model=model)
            checkpoint.restore(os.path.join(input_dir, f'run-{run_num}', f'ckpts-{EPOCH}')).expect_partial()
        else:
            model = tf.keras.models.load_model(os.path.join(input_dir, f'run-{run_num}', 'ckpts/model'))
        # create empty lists to store true and predicted classes
        pred_classes = []
        true_classes = []
        # create output file
        outfile = open(os.path.join(output_dir, f'testing-summary'), 'w')
        outfile.write(f'run: {run_num}\ntesting set: {set_type}\nnumber of classes: {NUM_CLASSES}\nvector size: {VECTOR_SIZE}\nvocabulary size: {VOCAB_SIZE}\nembedding size: {EMBEDDING_SIZE}\ndropout rate: {DROPOUT_RATE}\nbatch size per gpu: {BATCH_SIZE}\nglobal batch size: {BATCH_SIZE*hvd.size()}\nnumber of gpus: {hvd.size()}\n')

    # load testing tfrecords
    test_files = sorted(glob.glob(os.path.join(test_path, 'tfrecords', 'test-tfrec-*.tfrec')))
    test_idx_files = sorted(glob.glob(os.path.join(test_path, 'tfrecords', 'idx_files', 'test-tfrec-*.tfrec.idx')))

    num_preprocessing_threads = 4
    test_preprocessor = DALIPreprocessor(test_files, test_idx_files, BATCH_SIZE, num_preprocessing_threads, dali_cpu=True,
                                        deterministic=False, training=False)

    test_input = test_preprocessor.get_device_dataset()

    start = datetime.datetime.now()

    for batch, (reads, labels) in enumerate(test_input.take(test_steps), 1):
        batch_pred = testing_step(reads, labels, loss, test_loss, test_accuracy, model)

        if hvd.rank() == 0:
            pred_classes += [np.argmax(i) for i in batch_pred.numpy().tolist()]
            true_classes += labels.numpy().tolist()

    end = datetime.datetime.now()

    if hvd.rank() == 0:
        print(len(pred_classes), len(true_classes))
        # adjust list of predicted and true species
        if len(pred_classes) > num_reads:
            num_extra_reads = (test_steps*BATCH_SIZE) - num_reads
            pred_classes = pred_classes[:-num_extra_reads]
            true_classes = true_classes[:-num_extra_reads]
        print(len(pred_classes), len(true_classes))
        # create empty dictionary to store number of reads
        test_reads = defaultdict(int)
        for l in true_classes:
            test_reads[l] += 1
        # get precision and recall for each species
        list_labels = [class_mapping[str(i)] for i in range(len(class_mapping))]
        metrics_report(true_classes, pred_classes, list_labels, os.path.join(output_dir), class_mapping, test_reads, 'species', hvd.rank())
        # create list of colors for species
        # colors = get_colors('species', list_labels, input_dir)
        # ROCcurve(true_vectors, pred_vectors, class_mapping, output_dir, 'species', colors)
        # get results at other ranks
        # for r in ['genus', 'family', 'order', 'class']:
        #     # load dictionary mapping species labels to other ranks labels
        #     with open(os.path.join(input_dir, f'{r}_species_mapping_dict.json')) as f_json:
        #         rank_species_mapping = json.load(f_json)
        #     rank_pred_classes = [rank_species_mapping[str(i)] for i in pred_classes]
        #     rank_true_classes = [rank_species_mapping[str(i)] for i in true_classes]
        #     # get precision and recall for each class
        #     with open(os.path.join(input_dir, f'{r}_mapping_dict.json')) as f_json:
        #         rank_mapping = json.load(f_json)
        #     # get list of labels sorted
        #     rank_labels = [rank_mapping[str(i)] for i in range(len(rank_mapping))]
        #     # create dictionary to store number of reads
        #     rank_reads = defaultdict(int)
        #     for l in rank_true_classes:
        #         rank_reads[l] += 1
        #     metrics_report(rank_true_classes, rank_pred_classes, rank_labels, os.path.join(output_dir), rank_mapping, rank_reads, r, hvd.rank())
        #     r_colors = get_colors(r, rank_labels, input_dir)
        #     ROCcurve(true_vectors, pred_vectors, rank_mapping, output_dir, r, r_colors)



    total_time = end - start
    hours, seconds = divmod(total_time.seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    # get true and predicted classes
    outfile.write(f'Testing accuracy: {test_accuracy.result().numpy()*100}\t')
    outfile.write(f'Testing loss: {test_loss.result().numpy()}\t')
    outfile.write("Testing took %02d:%02d:%02d.%d\n" % (hours, minutes, seconds, total_time.microseconds))
    print("\nTesting took %02d:%02d:%02d.%d\n" % (hours, minutes, seconds, total_time.microseconds))



if __name__ == "__main__":
    main()
