import sys
import tensorflow as tf

def compute_loss(self, one_hot, predictions):
    loss = tf.reduce_sum(self.loss_object(one_hot, predictions)) * (1.0 / self.hparams.global_batch_size)
    return loss

@tf.function
# Define one train step
def train_step(self, inputs):
    reads, labels = inputs
    print(reads, labels)
    # Get loss
    with tf.GradientTape() as tape:
        # Get the predictions as probabilities
        predictions = self._model(reads, training=True)
        # Calculate the loss
        loss = compute_loss(self, labels, predictions)
        # compute the scaled loss: multiply the loss by the loss scale
#        scaled_loss = self.optimizer.get_scaled_loss(loss)
    # Compute the scaled gradients
#    scaled_gradients = tape.gradient(scaled_loss, self._model.trainable_variables)
    # compute the gradients correct values: take a list of scaled gradients as inputs and
    # divide each one by the loss scale to unscale them
#    gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
    gradients = tape.gradient(loss, self._model.trainable_variables)
    # Update the weights
    self.optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))
    # Update the accuracy
    self.train_accuracy.update_state(labels, predictions)
    return loss


@tf.function
def distributed_train_epoch(self, strategy):
    total_loss = 0.0
    num_train_batches = 0.0
    for one_batch in self.loaders['train']:
        #if num_train_batches >= 10 and num_train_batches < 20:
        #    with tf.profiler.experimental.Trace('train', step_num=one_batch):
        per_replica_loss = strategy.run(train_step, args=(self, one_batch,))
        total_loss_batch = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
        total_loss += total_loss_batch
         #   if num_train_batches == 20:
         #       tf.profiler.experimental.stop()
        num_train_batches += 1

    return total_loss, num_train_batches
