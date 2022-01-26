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
from models import AlexNet
from summarize import get_metrics, ROCcurve
import argparse

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
# comment next 2 lines if testing large dataset
# for gpu in gpus:
    # tf.config.experimental.set_memory_growth(gpu, True)
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
    start = datetime.datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecords', type=str, help='path to tfrecords', required=True)
    parser.add_argument('--dali_idx', type=str, help='path to dali indexes files', required=True)
    parser.add_argument('--class_mapping', type=str, help='path to json file containing dictionary mapping taxa to labels', required=True)
    parser.add_argument('--output_dir', type=str, help='directory to store results', required=True)
    parser.add_argument('--set_type', type=str, help='type of dataset', choices=['test', 'val', 'train'])
    # parser.add_argument('-tfrecord', type=str, help='tfrecord file to test')
    parser.add_argument('--epoch', type=int, help='epoch of checkpoint')
    parser.add_argument('--batch_size', type=int, help='batch size per gpu', default=512)
    parser.add_argument('--model', type=str, help='path to directory containing saved model')
    parser.add_argument('--ckpts', type=str, help='path to directory containing checkpoint file', required=('--epoch' and '--checkpoint' in sys.argv))
    parser.add_argument('--ckpt_prefix', type=str, help='prefix of checkpoint files', required=('--ckpts' in sys.argv))
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
    policy = keras.mixed_precision.Policy('mixed_float16')
    keras.mixed_precision.set_global_policy(policy)

    # define metrics
    loss = tf.losses.SparseCategoricalCrossentropy()
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    init_lr = 0.0001
    opt = tf.keras.optimizers.Adam(init_lr)
    opt = keras.mixed_precision.LossScaleOptimizer(opt)

    output_dir = os.path.join(args.output_dir, f'testing-{args.set_type}-set')

    if hvd.rank() == 0:
        # create output directory
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

    # load model
    if args.ckpts is not None:
        model = AlexNet(VECTOR_SIZE, EMBEDDING_SIZE, NUM_CLASSES, VOCAB_SIZE, DROPOUT_RATE)
        checkpoint = tf.train.Checkpoint(optimizer=opt, model=model)
        checkpoint.restore(os.path.join(args.ckpts, f'{args.ckpt_prefix}-{args.epoch}')).expect_partial()
    elif args.model is not None:
        model = tf.keras.models.load_model(args.model, 'model')
            # restore the checkpointed values to the model
    #        checkpoint = tf.train.Checkpoint(model)
    #        checkpoint.restore(tf.train.latest_checkpoint(os.path.join(input_dir, f'run-{run_num}', 'ckpts')))
    #        ckpt_path = os.path.join(input_dir, f'run-{run_num}', 'ckpts/ckpts')
    #        latest_ckpt = tf.train.latest_checkpoint(os.path.join(input_dir, f'run-{run_num}', 'ckpts'))
    #        print(f'latest ckpt: {latest_ckpt}')
    #        model.load_weights(os.path.join(input_dir, f'run-{run_num}', f'ckpts/ckpts-{epoch}'))

    # get list of testing tfrecords
    test_files = sorted(glob.glob(os.path.join(args.tfrecords, f'*{args.set_type}*.tfrec')))
    test_idx_files = sorted(glob.glob(os.path.join(args.dali_idx, f'*{args.set_type}*.idx')))
    num_reads_files = sorted(glob.glob(os.path.join(args.tfrecords, '*-read_count')))

    # split tfrecords between gpus
    test_files_per_gpu = math.ceil(len(test_files)/hvd.size())
    # gpu_test_files = test_files[hvd.rank()*test_files_per_gpu:(hvd.rank()+1)*test_files_per_gpu]
    # gpu_test_idx_files = test_idx_files[hvd.rank()*test_files_per_gpu:(hvd.rank()+1)*test_files_per_gpu]

    if hvd.rank() != hvd.size() - 1:
        gpu_test_files = test_files[hvd.rank()*test_files_per_gpu:(hvd.rank()+1)*test_files_per_gpu]
        gpu_test_idx_files = test_idx_files[hvd.rank()*test_files_per_gpu:(hvd.rank()+1)*test_files_per_gpu]
        gpu_num_reads_files = num_reads_files[hvd.rank()*test_files_per_gpu:(hvd.rank()+1)*test_files_per_gpu]
    else:
        gpu_test_files = test_files[hvd.rank()*test_files_per_gpu:len(test_files)]
        gpu_test_idx_files = test_idx_files[hvd.rank()*test_files_per_gpu:len(test_files)]
        gpu_num_reads_files = num_reads_files[hvd.rank()*test_files_per_gpu:len(test_files)]

    # create empty confusion matrix with rows = true classes and columns = predicted classes
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES))
    # record the number of correct and incorrect predictions
    num_correct_pred = 0
    num_incorrect_pred = 0
    num_reads_tested = 0
    # create summary file
    f_prob = open(os.path.join(output_dir, f'probabilities-{hvd.rank()}.tsv'), 'w')
    for i in range(len(gpu_test_files)):
        # get number of reads in test file
        with open(os.path.join(args.tfrecords, gpu_num_reads_files[i]), 'r') as f:
            num_reads = int(f.readlines()[0])
        num_reads_tested += num_reads
        # compute number of required steps to iterate over entire test file
        test_steps = math.ceil(num_reads/(args.batch_size))

        num_preprocessing_threads = 4
        test_preprocessor = DALIPreprocessor(gpu_test_files[i], gpu_test_idx_files[i], args.batch_size, num_preprocessing_threads, dali_cpu=True,
                                            deterministic=False, training=False)

        test_input = test_preprocessor.get_device_dataset()

        # create empty arrays to store the predicted and true values
        all_predictions = tf.zeros([args.batch_size, NUM_CLASSES], dtype=tf.dtypes.float32, name=None)
        all_labels = [tf.zeros([args.batch_size], dtype=tf.dtypes.float32, name=None)]

        for batch, (reads, labels) in enumerate(test_input.take(test_steps), 1):
            batch_predictions = testing_step(reads, labels, loss, test_loss, test_accuracy, model)
            if batch == 1:
                all_labels = [labels]
                all_predictions = batch_predictions
            else:
                all_predictions = tf.concat([all_predictions, batch_predictions], 0)
                all_labels = tf.concat([all_labels, [labels]], 1)

        # get list of true species, predicted species and predicted probabilities
        all_predictions = all_predictions.numpy()
        pred_species = [np.argmax(j) for j in all_predictions]
        pred_probabilities = [np.amax(j) for j in all_predictions]
        true_species = all_labels[0].numpy()
        true_one_hot = tf.one_hot(true_species, NUM_CLASSES)

        # save probabilities
        np.save(os.path.join(output_dir, f'true-probs-{hvd.rank()}-{i}'), true_one_hot)
        np.save(os.path.join(output_dir, f'pred-probs-{hvd.rank()}-{i}'), all_predictions)

        # adjust the list of predicted and true species if necessary
        if len(pred_species) > num_reads:
            num_extra_reads = (test_steps*args.batch_size) - num_reads
            pred_species = pred_species[:-num_extra_reads]
            true_species = true_species[:-num_extra_reads]

        # fill out confusion matrix and output probabilities
        for j in range(len(true_species)):
            cm[true_species[j], pred_species[j]] += 1
            if true_species[j] == pred_species[j]:
                num_correct_pred += 1
                f_prob.write(f'correct\t{true_species[j]}\t{pred_species[j]}\t{pred_probabilities[j]}\n')
            else:
                num_incorrect_pred += 1
                f_prob.write(f'incorrect\t{true_species[j]}\t{pred_species[j]}\t{pred_probabilities[j]}\n')

            # if hvd.rank() == 0:
                # pred_classes += [np.argmax(i) for i in batch_pred.numpy().tolist()]
                # true_classes += labels.numpy().tolist()

    f_prob.close()

    # write results to file
    with open(os.path.join(output_dir, f'confusion-matrix-{hvd.rank()}.tsv'), 'w') as cm_f:
        for i in range(NUM_CLASSES):
            cm_f.write(f'\t{class_mapping[str(i)]}')
        cm_f.write('\n')
        for i in range(NUM_CLASSES):
            cm_f.write(f'{class_mapping[str(i)]}')
            for j in range(NUM_CLASSES):
                cm_f.write(f'\t{cm[i,j]}')
            cm_f.write('\n')

    # compute average accuracy across all species
    accuracy = round(float(num_correct_pred)/(num_correct_pred + num_incorrect_pred), 5)

    end = datetime.datetime.now()
    total_time = end - start
    hours, seconds = divmod(total_time.seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    with open(os.path.join(output_dir, f'testing-summary-{hvd.rank()}.tsv'), 'w') as outfile:
        outfile.write(f'testing set: {args.set_type}\nnumber of classes: {NUM_CLASSES}\nvector size: {VECTOR_SIZE}\nvocabulary size: {VOCAB_SIZE}\nembedding size: {EMBEDDING_SIZE}\ndropout rate: {DROPOUT_RATE}\nbatch size per gpu: {args.batch_size}\nnumber of gpus: {hvd.size()}\nGPU: {hvd.rank()}\nnumber of tfrecord files tested: {len(gpu_test_files)}\nnumber of reads tested: {num_reads_tested}\n')

        if args.ckpts:
            outfile.write(f'checkpoint saved at epoch: {args.epoch}')
        else:
            outfile.write(f'model saved at last epoch')

        outfile.write(f'\naccuracy: {test_accuracy.result().numpy()*100}\naccuracy: {accuracy}\nloss: {test_loss.result().numpy()}\nrun time: {hours}:{minutes}:{seconds}:{total_time.microseconds}')


if __name__ == "__main__":
    main()
