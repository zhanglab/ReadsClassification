import tensorflow as tf

# Define one test step
def test_step(self, inputs):
    reads, labels = inputs
    predictions = self._model(reads, training=False)
    # Get true and predicted labels
    pred_classes = tf.keras.backend.argmax(predictions)
    t_classes = tf.keras.backend.argmax(labels)
    # Get highest probability in vector
    prob = tf.reduce_max(predictions, axis=1)
    # convert to proto tensor
    proto_tensor_t_classes = tf.make_tensor_proto(t_classes)
    proto_tensor_pred_classes = tf.make_tensor_proto(pred_classes)
    proto_tensor_prob = tf.make_tensor_proto(prob)
    self.true_classes += tf.make_ndarray(proto_tensor_t_classes).tolist()
    self.predicted_classes += tf.make_ndarray(proto_tensor_pred_classes).tolist()
    self.probabilities += tf.make_ndarray(proto_tensor_prob).tolist()

    test_loss = self.loss_object(labels, predictions)
    self.test_accuracy.update_state(labels, predictions)
    self.test_loss.update_state(test_loss)

def distributed_test_epoch(self, strategy):
    num_test_batches = 0
    for one_batch in self.loaders['test']:
        num_test_batches += 1
        strategy.experimental_run_v2(test_step, args=(self, one_batch,))

        if num_test_batches % 100 == 0:
            print('Test batch #: {}'.format(num_test_batches))
            print('accuracy: {}'.format(self.test_accuracy.result()))
            print('test loss: {}'.format(self.test_loss.result()))