import nvidia.dali as dali
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.plugin.tf as dali_tf
import nvidia.dali.tfrecord as tfrec

import tensorflow as tf

from collections import defaultdict
import os

from plots import learning_curves, create_barplot

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

def run_training(args):
    # reset tensorflow computation graph
    tf.compat.v1.reset_default_graph()

    # write down settings for training
    with open(os.path.join(args.output, args.model_name, 'training-info'), 'w+') as f:
        f.write(
            f'number of epochs: {args.epochs}\ndropout rate: {args.dropout_rate}\nbatch size: {args.batch_size}\n'
            f'global batch size: {args.global_batch_size}\nsteps per epoch: {args.steps_per_epoch}\nvalidation steps: {args.validation_steps}\n'
            f'reads in training set: {args.train_size}\nreads in validation set: {args.val_size}\n'
            f'read length: {args.read_length}\nk value: {args.k_value}\nnumber of devices: {args.num_gpu}\nnumber of classes: {args.num_classes}\n'
            f'embedding size: {args.embedding_size}\nnumber of kmers: {args.num_kmers}\ninput size: {args.vector_size}\n')

    # define shapes and types of data and labels
    shapes = (
        (args.batch_size, args.vector_size),
        (args.batch_size))
    dtypes = (
        tf.int64,
        tf.int64)

    # create callbacks
    list_callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(args.output, args.model_name, 'ckpts'), save_best_only=True, verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(args.output, args.model_name, 'logs'))
    ]

    # create an instance of strategy to perform synchronous training across multiple gpus
    strategy = tf.distribute.MirroredStrategy(devices=args.dict_gpus[args.num_gpu])

    with strategy.scope():
        # Define model to train: AlexNet
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(shape=(args.vector_size), dtype='int32'),
                tf.keras.layers.Embedding(input_dim=args.num_kmers + 1, output_dim=args.embedding_size,
                                          input_length=args.vector_size, trainable=True),
                tf.keras.layers.Reshape((args.vector_size, args.embedding_size, 1)),
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
                tf.keras.layers.Dropout(args.dropout_rate),
                tf.keras.layers.Dense(units=4096),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.Dropout(args.dropout_rate),
                tf.keras.layers.Dense(units=1000),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.Dropout(args.dropout_rate),
                tf.keras.layers.Dense(units=args.num_classes),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('softmax'),
            ]
        )

        model.compile(
           optimizer='adam',
           loss='sparse_categorical_crossentropy',
           metrics=['accuracy'])

        with open(os.path.join(args.output, args.model_name, 'model'), 'w+') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

        # distribute dataset
        def train_dataset_fn(input_context):
            with tf.device("/gpu:{}".format(input_context.input_pipeline_id)):
                device_id = input_context.input_pipeline_id
                return dali_tf.DALIDataset(fail_on_device_mismatch=False,
                    pipeline=TFRecordPipeline(args.batch_size, args.train_tfrecord, args.train_tfrecord_idx, device_id=device_id, shard_id=device_id, num_shards=args.num_gpu),
                    batch_size=args.batch_size, output_shapes=shapes, output_dtypes=dtypes, device_id=device_id)

        def val_dataset_fn(input_context):
            with tf.device("/gpu:{}".format(input_context.input_pipeline_id)):
                device_id = input_context.input_pipeline_id
                return dali_tf.DALIDataset(fail_on_device_mismatch=False,
                    pipeline=TFRecordPipeline(args.batch_size, args.val_tfrecord, args.val_tfrecord_idx, device_id=device_id, shard_id=device_id, num_shards=args.num_gpu),
                    batch_size=args.batch_size, output_shapes=shapes, output_dtypes=dtypes, device_id=device_id)

        input_options = tf.distribute.InputOptions(
                experimental_place_dataset_on_device=True,
                experimental_prefetch_to_device=False,
                experimental_replication_mode=tf.distribute.InputReplicationMode.PER_REPLICA)

        train_dataset = strategy.distribute_datasets_from_function(train_dataset_fn, input_options)
        val_dataset = strategy.distribute_datasets_from_function(val_dataset_fn, input_options)

        print("Fit model on training data")
        history = model.fit(train_dataset, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch, validation_data=val_dataset, validation_steps=args.validation_steps, callbacks=[list_callbacks])

        # create learning curves
        learning_curves(args, history)

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
        for i in range(len(train_true_classes)):
            train_dict_classes[train_true_classes[i]] += 1

        val_dict_classes = defaultdict(int)
        for i in range(len(val_true_classes)):
            val_dict_classes[val_true_classes[i]] += 1

        create_barplot(args, train_dict_classes, val_dict_classes)


if __name__ == "__main__":
    run_training()
