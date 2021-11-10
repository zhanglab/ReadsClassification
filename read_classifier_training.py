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
import random
from models import AlexNet, VGG16, VDCNN


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

@tf.function
def training_step(reads, labels, train_accuracy, loss, opt, model, first_batch):
    with tf.GradientTape() as tape:
        probs = model(reads, training=True)
        # get the loss
        loss_value = loss(labels, probs)
        # scale the loss (multiply the loss by a factor) to avoid numeric underflow
        scaled_loss = opt.get_scaled_loss(loss_value)
    # use DistributedGradientTape to wrap tf.GradientTape and use an allreduce to
    # combine gradient values before applying gradients to model weights
    tape = hvd.DistributedGradientTape(tape)
    # get the scaled gradients
    scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
    # get the unscaled gradients
    grads = opt.get_unscaled_gradients(scaled_gradients)
    #grads = tape.gradient(loss_value, model.trainable_variables)
    #opt.apply_gradients(zip(grads, model.trainable_variables))
    opt.apply_gradients(zip(grads, model.trainable_variables))
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    # Note: broadcast should be done after the first gradient step to ensure optimizer
    # initialization.
    if first_batch:
        print(f'First_batch: {first_batch}')
        hvd.broadcast_variables(model.variables, root_rank=0)
        hvd.broadcast_variables(opt.variables(), root_rank=0)

    #update training accuracy
    train_accuracy.update_state(labels, probs)

    return loss_value, grads

@tf.function
def testing_step(reads, labels, loss, val_loss, val_accuracy, model):
    probs = model(reads, training=False)
    val_accuracy.update_state(labels, probs)
    loss_value = loss(labels, probs)
    val_loss.update_state(loss_value)

def main():
    input_dir = sys.argv[1]
    run_num = sys.argv[2]
    # define some training and model parameters
    EPOCHS = int(sys.argv[3])
    VECTOR_SIZE = 250 - 12 + 1
    VOCAB_SIZE = 8390657
    EMBEDDING_SIZE = 60
    DROPOUT_RATE = float(sys.argv[4])
    BATCH_SIZE = int(sys.argv[5])
    num_train_samples = int(sys.argv[6])
    num_val_samples = int(sys.argv[7])
    init_lr = float(sys.argv[8])
    lr_decay = int(sys.argv[9])
    print(f'{hvd.rank()}/{hvd.local_rank()} # train samples: {num_train_samples}')
    print(f'{hvd.rank()}/{hvd.local_rank()} # val samples: {num_val_samples}')

    # load class_mapping file mapping label IDs to species
    f = open(os.path.join(input_dir, 'class_mapping.json'))
    class_mapping = json.load(f)
    NUM_CLASSES = len(class_mapping)

    # create dtype policy
    policy = keras.mixed_precision.Policy('mixed_float16')
    keras.mixed_precision.set_global_policy(policy)
    print('Compute dtype: %s' % policy.compute_dtype)
    print('Variable dtype: %s' % policy.variable_dtype)

    # load training and validation tfrecords
    train_files = random.shuffle(glob.glob(os.path.join(input_dir, 'tfrecords', f'*-train*.tfrec')))
    print(train_files)
    train_files_ids = ['-'.join(i.split('-')[:3]) if len(i.split('-')) == 4 else '-'.join(i.split('-')[:2]) for i in train_files]
    train_idx_files = [os.path.join(input_dir, 'tfrecords', 'idx_files', f'{i}-reads.tfrec.idx') for i in train_files_ids]
    val_files = random.shuffle(glob.glob(os.path.join(input_dir, 'tfrecords', f'*-val*.tfrec')))
    val_files_ids = ['-'.join(i.split('-')[:3]) if len(i.split('-')) == 4 else '-'.join(i.split('-')[:2]) for i in val_files]
    val_idx_files = [os.path.join(input_dir, 'tfrecords', 'idx_files', f'{i}-reads.tfrec.idx') for i in val_files_ids]
    print(f'{hvd.rank()}/{hvd.local_rank()} # train files: {len(train_files)}\t{len(train_idx_files)}\t{train_files}')
    print(f'{hvd.rank()}/{hvd.local_rank()} # val files: {len(val_files)}\t{len(val_idx_files)}\t{val_files}')

    nstep_per_epoch = num_train_samples // (BATCH_SIZE*hvd.size())
    print(f'{hvd.rank()}/{hvd.local_rank()} # steps per epoch for whole train dataset: {nstep_per_epoch}')
    val_steps = num_val_samples // (BATCH_SIZE*hvd.size())
    print(f'{hvd.rank()}/{hvd.local_rank()} # steps for whole val dataset: {val_steps}')

    num_preprocessing_threads = 4
    train_preprocessor = DALIPreprocessor(train_files, train_idx_files, BATCH_SIZE, num_preprocessing_threads, VECTOR_SIZE,
                                          dali_cpu=True, deterministic=False, training=True)
    val_preprocessor = DALIPreprocessor(val_files, val_idx_files, BATCH_SIZE, num_preprocessing_threads, VECTOR_SIZE, dali_cpu=True,
                                        deterministic=False, training=False)

    train_input = train_preprocessor.get_device_dataset()
    val_input = val_preprocessor.get_device_dataset()

    # define model
    model = AlexNet(input_dir, VECTOR_SIZE, EMBEDDING_SIZE, NUM_CLASSES, VOCAB_SIZE, DROPOUT_RATE, run_num)

    # define metrics
    loss = tf.losses.SparseCategoricalCrossentropy()
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

    # define initial learning rate
    opt = tf.keras.optimizers.Adam(init_lr)
    opt = keras.mixed_precision.LossScaleOptimizer(opt)
    output_dir = os.path.join(input_dir, f'run-{run_num}')

    if hvd.rank() == 0:
        # create output directory
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        # create directory for storing checkpoints
        ckpt_dir = os.path.join(output_dir, 'ckpts')
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        checkpoint = tf.train.Checkpoint(model=model, optimizer=opt)

        # create directory for storing logs
        tensorboard_dir = os.path.join(output_dir, 'logs')
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)

        writer = tf.summary.create_file_writer(tensorboard_dir)
        conv_summary_writer = tf.summary.create_file_writer(tensorboard_dir)
        emb_summary_writer = tf.summary.create_file_writer(tensorboard_dir)
        dense_summary_writer = tf.summary.create_file_writer(tensorboard_dir)
        grads_summary_writer = tf.summary.create_file_writer(tensorboard_dir)

        # create summary file
        with open(os.path.join(output_dir, 'training-summary'), 'w') as f:
            f.write(f'Run: {run_num}\nNumber of classes: {NUM_CLASSES}\nEpochs: {EPOCHS}\nVector size: {VECTOR_SIZE}\nVocabulary size: {VOCAB_SIZE}\nEmbedding size: {EMBEDDING_SIZE}\nDropout rate: {DROPOUT_RATE}\nBatch size per gpu: {BATCH_SIZE}\nGlobal batch size: {BATCH_SIZE*hvd.size()}\nNumber of gpus: {hvd.size()}\nTraining set size: {num_train_samples}\nValidation set size: {num_val_samples}\nNumber of steps per epoch for each process: {nstep_per_epoch*hvd.size()}\nNumber of stepsfor each process and for whole validation dataset: {val_steps*hvd.size()}\nInitial learning rate: {init_lr}\nLearning rate decay: every {lr_decay} epochs')


    start = datetime.datetime.now()
    epoch = 1

    for batch, (reads, labels) in enumerate(train_input.take(nstep_per_epoch*EPOCHS), 1):
        print(f'{hvd.rank()}\t{labels}')
        if batch == 1000:
            break
        # get training loss
        # loss_value, gradients = training_step(reads, labels, train_accuracy, loss, opt, model, batch == 1)
        # if batch % 100 == 0 and hvd.rank() == 0:
        #     print(f'GPU: {hvd.rank()}/{hvd.local_rank()} - Step {batch} - Training loss: {loss_value} - Training accuracy: {train_accuracy.result().numpy()*100}')
        #     print(f'Epoch: {epoch} - Step: {batch} - learning rate: {opt.learning_rate}')
        #     # write metrics
        #     with writer.as_default():
        #         tf.summary.scalar("learning_rate", opt.learning_rate, step=batch)
        #         tf.summary.scalar("train_loss", loss_value, step=batch)
        #         tf.summary.scalar("train_accuracy", train_accuracy.result().numpy(), step=batch)
        #         writer.flush()
        #
        # # evaluate model at the end of every epoch
        # if batch % nstep_per_epoch == 0:
        #     for _, (reads, labels) in enumerate(val_input.take(val_steps)):
        #         testing_step(reads, labels, loss, val_loss, val_accuracy, model)
        #
        #     # adjust learning rate
        #     if epoch % lr_decay == 0:
        #         current_lr = opt.learning_rate
        #         new_lr = current_lr / 2
        #         opt.learning_rate = new_lr
        #
        #     if hvd.rank() == 0:
        #         print(f'Process {hvd.local_rank()}/{hvd.rank()} - Epoch: {epoch} - Step: {batch} - Validation loss: {val_loss.result().numpy()} - Validation accuracy: {val_accuracy.result().numpy()*100}')
        #         # save weights
        #         checkpoint.save(ckpt_dir)
        #         model.save(os.path.join(ckpt_dir, 'model'))
        #         with writer.as_default():
        #             tf.summary.scalar("val_loss", val_loss.result().numpy(), step=epoch)
        #             tf.summary.scalar("val_accuracy", val_accuracy.result().numpy(), step=epoch)
        #             writer.flush()
        #         # save embedding weights
        #         emb_weights = model.get_layer('embedding').get_weights()
        #         with emb_summary_writer.as_default():
        #             tf.summary.histogram('embeddings', emb_weights[0], step=epoch)
        #             emb_summary_writer.flush()
        #         # save C1 layer weights
        #         weights_conv = model.get_layer('conv_1').get_weights()
        #         with conv_summary_writer.as_default():
        #             tf.summary.histogram('conv_1/weights', weights_conv[0], step=epoch)
        #             tf.summary.histogram('conv_1/bias', weights_conv[1], step=epoch)
        #             conv_summary_writer.flush()
        #         # save output layer weights
        #         weights_dense = model.get_layer('last_dense').get_weights()
        #         with dense_summary_writer.as_default():
        #             tf.summary.histogram('last_dense/weights', weights_dense[0], step=epoch)
        #             tf.summary.histogram('last_dense/bias', weights_dense[1], step=epoch)
        #             dense_summary_writer.flush()
        #         # save gradients
        #         with grads_summary_writer.as_default():
        #             curr_grad = gradients[0]
        #             tf.summary.histogram('grad_histogram', curr_grad, step=epoch)
        #             grads_summary_writer.flush()
        #
        #     # reset metrics variables
        #     val_loss.reset_states()
        #     train_accuracy.reset_states()
        #     val_accuracy.reset_states()

            # define end of current epoch
            epoch += 1

    # if hvd.rank() == 0:
    #     # save final embeddings
    #     emb_weights = model.get_layer('embedding').get_weights()[0]
    #     out_v = io.open(os.path.join(output_dir, 'embeddings.tsv'), 'w', encoding='utf-8')
    #     print(f'# embeddings: {len(emb_weights)}')
    #     for i in range(len(emb_weights)):
    #         vec = emb_weights[i]
    #         out_v.write('\t'.join([str(x) for x in vec]) + "\n")
    #     out_v.close()

    end = datetime.datetime.now()

    if hvd.rank() == 0:
        total_time = end - start
        hours, seconds = divmod(total_time.seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        with open(os.path.join(output_dir, 'training-summary'), 'a') as f:
            f.write("\nTook %02d:%02d:%02d.%d\n" % (hours, minutes, seconds, total_time.microseconds))
        print("\nTook %02d:%02d:%02d.%d\n" % (hours, minutes, seconds, total_time.microseconds))

if __name__ == "__main__":
    main()
