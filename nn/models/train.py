import sys
import tensorflow as tf

def compute_loss(self, label, predictions):
    loss = tf.reduce_sum(self.loss_object(label, predictions)) * (
                1.0 / (self.hparams.global_batch_size))
    return loss

# Define one train step
def train_step(self, inputs):
    reads, labels = inputs
    with tf.GradientTape() as tape:
        # Get the predictions as probabilities
        predictions = self._model(reads, training=True)
        # Calculate the loss
        loss = compute_loss(self, labels, predictions)
    # Compute the gradients
    gradients = tape.gradient(loss, self._model.trainable_variables)
    # Update the weights
    self.optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))
    # Update the accuracy
    self.train_accuracy.update_state(labels, predictions)
    return loss


def distributed_train_epoch(self, strategy):
    total_loss = 0.0
    num_train_batches = 0.0
    for one_batch in self.loaders['train']:
        # Get the loss from each GPU/device
        per_replica_loss = strategy.experimental_run_v2(train_step, args=(self, one_batch,))
        total_loss_batch = strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None).numpy()
        total_loss += total_loss_batch
        num_train_batches += 1

    return total_loss, num_train_batches

