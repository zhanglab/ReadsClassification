import nvidia.dali as dali
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.plugin.tf as dali_tf
import nvidia.dali.tfrecord as tfrec

import tensorflow as tf
from tensorflow import keras

import json
from collections import defaultdict
import os
import sys
import io

from nn.dali_nn.dali_reads_classifier.plots import learning_curves, create_barplot

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

f = open(os.path.join(os.getcwd(), 'class_mapping.json'))
class_mapping = json.load(f)

FOLD = int(sys.argv[1])
tfrecords_path = str(sys.argv[2])
K_VALUE = 7
READ_LENGTH = 250
WINDOW_SIZE = 5
NUM_CLASSES = len(class_mapping)
NUM_DEVICES = 4      # number of GPUs
BATCH_SIZE = 125  # batch size per GPU
GLOBAL_BATCH_SIZE = NUM_DEVICES*BATCH_SIZE
VOCAB_SIZE = 4**K_VALUE
EPOCHS = 100
EMBEDDING_SIZE = 150
DROPOUT_RATE = 0.5
NUM_NS = 4
VECTOR_SIZE = READ_LENGTH - K_VALUE + 1
TRAINING_SIZE = 586484
VALIDATION_SIZE = 146470
STEPS_PER_EPOCH = math.ceil(TRAINING_SIZE / GLOBAL_BATCH_SIZE)
VALIDATION_STEPS = math.ceil(VALIDATION_SIZE / GLOBAL_BATCH_SIZE)

train_tfrecord = os.path.join(tfrecords_path, 'training_data_fold{FOLD}.tfrec')
train_tfrecord_idx = os.path.join(tfrecords_path, 'idx_files/training_data_fold{FOLD}.tfrec.idx')
val_tfrecord = os.path.join(tfrecords_path, 'validation_data_fold{FOLD}.tfrec')
val_tfrecord_idx = os.path.join(tfrecords_path, 'idx_files/validation_data_fold{FOLD}.tfrec.idx')

# write down settings for training
f = open(os.path.join(args.output, 'cross-validation-info'), 'a')
f.write(f'Training of model with fold # {FOLD}\n'
        f'batch size: {BATCH_SIZE}\n'
        f'global batch size: {GLOBAL_BATCH_SIZE}\nsteps per epoch: {STEPS_PER_EPOCH}\nvalidation steps: {VALIDATION_STEPS}\n'
        f'reads in training set: {TRAINING_SIZE}\nreads in validation set: {VALIDATION_SIZE}\n'
        f'read length: {READ_LENGTH}\nk value: {K_VALUE}\nnumber of devices: {NUM_DEVICES}\nnumber of classes: {NUM_CLASSES}\n'
        f'embedding size: {EMBEDDING_SIZE}\ndropout rate: {DROPOUT_RATE}\nnumber of kmers: {VOCAB_SIZE}\ninput size: {VECTOR_SIZE}\nwindow size: {WINDOW_SIZE}\n'
        f'number of negative samples: {NUM_NS}')

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

# create callbacks
list_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=1),
    tf.keras.callbacks.TensorBoard(log_dir=f'./logs-fold{i}')
]

class EarlyStoppingAtMinLoss(keras.callbacks.Callback):
    """Stop training when the validation loss is at its min, i.e. the validation loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

    def __init__(self, patience=5):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("val_loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))



class TFRecordPipeline(Pipeline):
    def __init__(self, batch_size, tfrecord, tfrecord_idx, device_id=0, shard_id=0, num_shards=1, num_threads=4, seed=0):
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

# load kmer vocabulary
f = open(os.path.join(os.getcwd(), 'kmer-vocab'), 'r')
content = f.readlines()
vocab = [i.rstrip().split('\t')[0] for i in content]
# load embeddings
embedding_matrix = np.zeros((VOCAB_SIZE+1, EMBEDDING_SIZE))
with io.open(os.path.join(os.getcwd(), 'vectors.tsv'), 'r', encoding='utf-8') as f:
    num_lines = 0
    for index, line in enumerate(f):
        vec = np.asarray(line.rstrip(), dtype='float32')
        embedding_matrix[index] = vec
        num_lines += 1
print(f'number of vectors saved in vectors.tsv: {num_lines} - vocabulary size: {VOCAB_SIZE}')

# create an instance of strategy to perform synchronous training across multiple gpus
strategy = tf.distribute.MirroredStrategy(devices=['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3'])

with strategy.scope():
    # Define model to train: AlexNet
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(VECTOR_SIZE), dtype='int32'),
            tf.keras.layers.Embedding(input_dim=VOCAB_SIZE+1, output_dim=EMBEDDING_SIZE,
                                      input_length=VECTOR_SIZE, embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                                      mask_zero=True, trainable=False),
            tf.keras.layers.Reshape((VECTOR_SIZE, EMBEDDING_SIZE, 1)),
            tf.keras.layers.Conv2D(96, kernel_size=(11, 11), strides=(4, 4), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
            tf.keras.layers.Conv2D(256, kernel_size=(5, 5), strides=(1, 1), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
            tf.keras.layers.Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=4096),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(DROPOUT_RATE),
            tf.keras.layers.Dense(units=4096),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(DROPOUT_RATE),
            tf.keras.layers.Dense(units=1000),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(DROPOUT_RATE),
            tf.keras.layers.Dense(NUM_CLASSES),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('softmax'),
        ]
    )

    model.compile(
       optimizer='adam',
       loss='sparse_categorical_crossentropy',
       metrics=['accuracy'])

    # only write down summary of model once
    with open(os.path.join(os.getcwd(), 'model'), 'w+') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    # distribute dataset
    def train_dataset_fn(input_context):
        with tf.device("/gpu:{}".format(input_context.input_pipeline_id)):
            device_id = input_context.input_pipeline_id
            return dali_tf.DALIDataset(fail_on_device_mismatch=False,
                pipeline=TFRecordPipeline(BATCH_SIZE, train_tfrecord, train_tfrecord_idx, device_id=device_id, shard_id=device_id, num_shards=NUM_DEVICES),
                batch_size=BATCH_SIZE, output_shapes=shapes, output_dtypes=dtypes, device_id=device_id)

    def val_dataset_fn(input_context):
        with tf.device("/gpu:{}".format(input_context.input_pipeline_id)):
            device_id = input_context.input_pipeline_id
            return dali_tf.DALIDataset(fail_on_device_mismatch=False,
                pipeline=TFRecordPipeline(BATCH_SIZE, val_tfrecord, val_tfrecord_idx, device_id=device_id, shard_id=device_id, num_shards=NUM_DEVICES),
                batch_size=BATCH_SIZE, output_shapes=shapes, output_dtypes=dtypes, device_id=device_id)

    input_options = tf.distribute.InputOptions(
            experimental_place_dataset_on_device=True,
            experimental_prefetch_to_device=False,
            experimental_replication_mode=tf.distribute.InputReplicationMode.PER_REPLICA)

    train_dataset = strategy.distribute_datasets_from_function(train_dataset_fn, input_options)
    val_dataset = strategy.distribute_datasets_from_function(val_dataset_fn, input_options)

    print("Fit model on training data")
    history = model.fit(train_dataset, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, validation_data=val_dataset, validation_steps=VALIDATION_STEPS, callbacks=[list_callbacks])

    # create learning curves
    learning_curves(args, history, f'LearningCurves-fold-{FOLD}.png')

    # get true labels
    def get_labels(inputs):
        _, labels = inputs
        t_classes = labels.numpy()
        return t_classes.tolist()

    def get_true_labels(dataset, num_steps):
        local_num_steps = 0
        true_classes = []
        for inputs in dataset:
            local_num_steps += 1
            true_classes += strategy.run(get_labels, args=(inputs,))
            if local_num_steps == num_steps:
                break

        return true_classes

    train_true_classes = get_true_labels(train_dataset, args.steps_per_epoch)
    val_true_classes = get_true_labels(val_dataset, args.validation_steps)

    train_dict_classes = defaultdict(int)
    for j in range(len(train_true_classes)):
        train_dict_classes[train_true_classes[j]] += 1

    val_dict_classes = defaultdict(int)
    for j in range(len(val_true_classes)):
        val_dict_classes[val_true_classes[j]] += 1

    create_barplot(args, train_dict_classes, val_dict_classes, f'data-barplots-fold-{FOLD}')

f.close()

