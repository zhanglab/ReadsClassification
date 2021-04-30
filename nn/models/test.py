import tensorflow as tf
from .utils import load_json_file

@tf.function
# Define one test step
def test_step_training(self, inputs):
    # in testing and training mode the labels are actual labels (integers),
    # in inference mode the labels are read ids
    reads, labels = inputs
    predictions = self._model(reads, training=False)
    # update test accuracy for each taxonomic rank
    self.test_accuracy.update_state(labels, predictions)
    # get test loss
    loss = self.loss_object(labels, predictions)
    # update test loss
    self.test_loss.update_state(loss)

@tf.function
def test_step_testing(self, inputs):
    # in inference mode the labels are read ids and in testing mode the labels
    # are integers assigned to the genomes in the testing set (different from integers mapping classes in training set)
    reads, labels = inputs
    predictions = self._model(reads, training=False)
    pred_classes = tf.keras.backend.argmax(predictions)
    self.predicted_classes.append(pred_classes.numpy())
    self.predictions.append(predictions.numpy())
    self.read_ids.append(labels.numpy())

@tf.function
def distributed_test_epoch(self, strategy):
    functions_dict = {'testing': test_step_testing, 'training': test_step_training,
                      'inference': test_step_testing}
    num_test_batches = 0.0
    for one_batch in self.loaders['test']:
        strategy.run(functions_dict[self.hparams.mode], args=(self, one_batch,))
        num_test_batches += 1
    return num_test_batches
