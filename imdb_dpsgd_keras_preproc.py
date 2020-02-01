# Copyright 2019, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Training a CNN on MNIST with Keras and the DP SGD optimizer."""




from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds

from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer

GradientDescentOptimizer = tf.compat.v1.train.GradientDescentOptimizer

flags.DEFINE_boolean(
    'dpsgd', True, 'If True, train with DP-SGD. If False, '
    'train with vanilla SGD.')
flags.DEFINE_float('learning_rate', 0.15, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 1.1,
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 250, 'Batch size')
flags.DEFINE_integer('epochs', 60, 'Number of epochs')
flags.DEFINE_integer(
    'microbatches', 250, 'Number of microbatches '
    '(must evenly divide batch_size)')
flags.DEFINE_string('model_dir', None, 'Model directory')

FLAGS = flags.FLAGS


def compute_epsilon(steps):
  """Computes epsilon value for given hyperparameters."""
  if FLAGS.noise_multiplier == 0.0:
    return float('inf')
  orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
  sampling_probability = FLAGS.batch_size / 60000
  rdp = compute_rdp(q=sampling_probability,
                    noise_multiplier=FLAGS.noise_multiplier,
                    steps=steps,
                    orders=orders)
  # Delta is set to 1e-5 because MNIST has 60000 training points.
  return get_privacy_spent(orders, rdp, target_delta=1e-5)[0]

def main(unused_argv):
  logging.set_verbosity(logging.INFO)




  if FLAGS.dpsgd and FLAGS.batch_size % FLAGS.microbatches != 0:
    raise ValueError('Number of microbatches should divide evenly batch_size')

  (train_data, test_data), info = tfds.load(
      # Use the version pre-encoded with an ~8k vocabulary.
      'imdb_reviews/subwords8k',
      # Return the train/test datasets as a tuple.
      split=(tfds.Split.TRAIN, tfds.Split.TEST),
      # Return (example, label) pairs from the dataset (instead of a dictionary).
      as_supervised=True,
      # Also return the `info` structure.
      with_info=True)

  encoder = info.features['text'].encoder
  print('Vocabulary size: {}'.format(encoder.vocab_size))
  sample_string = 'Hello TensorFlow.'

  encoded_string = encoder.encode(sample_string)
  print('Encoded string is {}'.format(encoded_string))

  original_string = encoder.decode(encoded_string)
  print('The original string: "{}"'.format(original_string))

  assert original_string == sample_string

  for ts in encoded_string:
      print('{} ----> {}'.format(ts, encoder.decode([ts])))

  for train_example, train_label in train_data.take(1):
      print('Encoded text:', train_example[:10].numpy())
      print('Label:', train_label.numpy())

  BUFFER_SIZE = 1000

  train_batches = (
      train_data
          .shuffle(BUFFER_SIZE)
          .padded_batch(32))

  test_batches = (
      test_data
          .padded_batch(32))

  for example_batch, label_batch in train_batches.take(2):
      print("Batch shape:", example_batch.shape)
      print("label shape:", label_batch.shape)

  model = tf.keras.Sequential([
      tf.keras.layers.Embedding(encoder.vocab_size, 16),
      tf.keras.layers.GlobalAveragePooling1D(),
      tf.keras.layers.Dense(1, activation='sigmoid')])


  model.summary()

  optimizer = DPGradientDescentGaussianOptimizer(
      l2_norm_clip=FLAGS.l2_norm_clip,
      noise_multiplier=FLAGS.noise_multiplier,
      num_microbatches=FLAGS.microbatches,
      learning_rate=FLAGS.learning_rate)
  # Compute vector of per-example loss rather than its mean over a minibatch.
  loss = tf.keras.losses.BinaryCrossentropy(
      from_logits=True)

  model.compile(optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy'])

  history = model.fit(train_batches,
                      epochs=10,
                      validation_data=test_batches,
                      validation_steps=30)

  loss, accuracy = model.evaluate(test_batches)

  print("Loss: ", loss)
  print("Accuracy: ", accuracy)

  # Compute the privacy budget expended.
  if FLAGS.dpsgd:
    eps = compute_epsilon(FLAGS.epochs * 60000 // FLAGS.batch_size)
    print('For delta=1e-5, the current epsilon is: %.2f' % eps)
  else:
    print('Trained with vanilla non-private SGD optimizer')

if __name__ == '__main__':
  app.run(main)
