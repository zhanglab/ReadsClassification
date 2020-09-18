import tensorflow as tf

# Define one test step
def test_step(self, inputs):
    reads, labels = inputs
    predictions = self._model(reads, training=False)
    # Get true and predicted labels
#    pred_classes = tf.keras.backend.argmax(predictions)
#    t_classes = tf.keras.backend.argmax(labels)
#    proto_tensor_t_classes = tf.make_tensor_proto(t_classes)
#    proto_tensor_pred_classes = tf.make_tensor_proto(pred_classes)
#    self.true_classes += tf.make_ndarray(proto_tensor_t_classes).tolist()
#    self.predicted_classes += tf.make_ndarray(proto_tensor_pred_classes).tolist()

    self.test_accuracy.update_state(labels, predictions)
    self.test_loss.update_state(self.loss_object(labels, predictions))

def distributed_test_epoch(self, strategy):
    for one_batch in self.loaders['val']:
        strategy.run(test_step, args=(self, one_batch,))
