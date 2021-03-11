import nvidia.dali as dali
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.plugin.tf as dali_tf
import nvidia.dali.tfrecord as tfrec

import tensorflow as tf
import numpy as np

import os

class TFRecordPipeline(Pipeline):
    def __init__(self, batch_size, tfrecord, tfrecord_idx, device_id=0, shard_id=0, num_shards=1, num_threads=4, seed=0):
        super(TFRecordPipeline, self).__init__(batch_size, num_threads, device_id, seed)
        self.input = ops.TFRecordReader(path=tfrecord, random_shuffle=False, shard_id=shard_id, num_shards=num_shards,
                                        index_path=tfrecord_idx,
                                        features={"read": tfrec.VarLenFeature([], tfrec.int64, 0),
                                                "label": tfrec.FixedLenFeature([1], tfrec.int64, -1)})

    def define_graph(self):
        inputs = self.input()
        reads = inputs["read"]
        labels = inputs["label"]
        labels = labels.gpu()
        return (reads, labels)


def run_testing(args):
    # reset tensorflow computation graph
    tf.compat.v1.reset_default_graph()

    # write down settings for testing
    with open(os.path.join(args.output, args.model_name, 'testing-info'), 'w+') as f:
        f.write(
            f'batch size: {args.batch_size}\nglobal batch size: {args.global_batch_size}\ntesting steps: {args.testing_steps}\n'
            f'reads in testing set: {args.test_size}\nnumber of devices: {args.num_gpu}\n')

    # define shapes and types of data and labels
    shapes = (
        (args.batch_size, args.vector_size),
        (args.batch_size))
    dtypes = (
        tf.int64,
        tf.int64)

    # create an instance of strategy to perform synchronous training across multiple gpus
    strategy = tf.distribute.MirroredStrategy(devices=args.dict_gpus[args.num_gpu])

    with strategy.scope():
        # load model
        model = tf.keras.models.load_model(os.path.join(args.input, args.model_name, 'ckpts'))
        print('model loaded')

        def test_dataset_fn(input_context):
            with tf.device("/gpu:{}".format(input_context.input_pipeline_id)):
                device_id = input_context.input_pipeline_id
                return dali_tf.DALIDataset(fail_on_device_mismatch=False,
                    pipeline=TFRecordPipeline(args.batch_size, args.test_tfrecord, args.test_tfrecord_idx, device_id=device_id, shard_id=device_id, num_shards=args.num_gpu),
                    batch_size=args.batch_size, output_shapes=shapes, output_dtypes=dtypes, device_id=device_id)

        input_options = tf.distribute.InputOptions(
                experimental_place_dataset_on_device=True,
                experimental_prefetch_to_device=False,
                experimental_replication_mode=tf.distribute.InputReplicationMode.PER_REPLICA)


        test_dataset = strategy.distribute_datasets_from_function(test_dataset_fn, input_options)
        print("Make predictions on testing data")
        model.evaluate(test_dataset, steps=args.testing_steps)

#        predictions = model.predict(test_dataset, steps=args.testing_steps)

#        print(tf.executing_eagerly())
#        # get true labels
#        def get_labels(inputs):
#            _, labels = inputs
#            t_classes = labels.numpy()
#            return t_classes.tolist()

#        local_num_steps = 0
#        for inputs in test_dataset:
#            local_num_steps += 1
#            t_classes = strategy.run(get_labels, args=(inputs,))
#            print(t_classes)
#            test_true_classes += t_classes
#            if local_num_steps == args.testing_steps:
#                break

        # compute testing accuracy
#        pred_list = list(predictions)
#        print(len(pred_list), len(test_true_classes))
#        correct_pred = 0
#        incorrect_pred = 0
#        for i in range(len(pred_list)):
#            predicted_class = np.argmax(pred_list[i])
#            if predicted_class == test_true_classes[i]:
#                correct_pred += 1
#            else:
#                incorrect_pred += 1
#        print(f'total number of reads tested: {len(pred_list)} - correct predictions: {correct_pred} - incorrect predictions: {incorrect_pred}')
#        print(f'Accuracy: {round(correct_pred/len(pred_list), 5)}')
#        with open(os.path.join(args.output, 'testing-info'), 'w+') as f:
#            f.write(f'Number of reads tested: {len(pred_list)}\nTesting accuracy: {round(correct_pred/len(pred_list), 5)}')   

if __name__ == "__main__":
    run_testing()
