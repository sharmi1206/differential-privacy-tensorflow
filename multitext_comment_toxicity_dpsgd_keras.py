import tensorflow as tf
print(tf.__version__)

import numpy as np
import pandas as pd

from absl import app
from absl import flags
from absl import logging
import os


TRAIN_DATA = "data/comments_train.csv"
TEST_DATA = "data/comments_test.csv"
TEST_LABELS = "data/test_labels.csv"
GLOVE_EMBEDDING = "embedding/glove.6B/glove.6B.100d.txt"

train = pd.read_csv(TRAIN_DATA)
test = pd.read_csv(TEST_DATA)
y_true = pd.read_csv(TEST_LABELS)
test = (pd.merge(test, y_true, on='id'))

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

max_words = 100000
max_len = 150
embed_size = 100


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
    train["comment_text"].fillna("fillna")
    test["comment_text"].fillna("fillna")


    x_train = train["comment_text"].str.lower()
    y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values

    x_test = train["comment_text"].str.lower()
    y_test = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words, lower=True)
    tokenizer.fit_on_texts(x_train)
    x_train = tokenizer.texts_to_sequences(x_train)
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)


    embeddings_index = {}

    with open(GLOVE_EMBEDDING, encoding='utf8') as f:
        for line in f:
            values = line.rstrip().rsplit(' ')
            word = values[0]
            embed = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embed

    word_index = tokenizer.word_index

    num_words = min(max_words, len(word_index) + 1)

    embedding_matrix = np.zeros((num_words, embed_size), dtype='float32')

    for word, i in word_index.items():

        if i >= max_words:
            continue

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


    input = tf.keras.layers.Input(shape=(max_len,))

    x = tf.keras.layers.Embedding(max_words, embed_size, weights=[embedding_matrix], trainable=False)(input)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True, dropout=0.1,
                                                          recurrent_dropout=0.1))(x)

    x = tf.keras.layers.Conv1D(64, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform")(x)

    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)

    x = tf.keras.layers.concatenate([avg_pool, max_pool])

    preds = tf.keras.layers.Dense(6, activation="sigmoid")(x)

    model = tf.keras.Model(input, preds)

    model.summary()
    if FLAGS.dpsgd:
        optimizer = DPGradientDescentGaussianOptimizer(
            l2_norm_clip=FLAGS.l2_norm_clip,
            noise_multiplier=FLAGS.noise_multiplier,
            num_microbatches=FLAGS.microbatches,
            learning_rate=FLAGS.learning_rate)
        # Compute vector of per-example loss rather than its mean over a minibatch.
        loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=True)  # reduction=tf.compat.v1.losses.Reduction.NONE
    else:
        optimizer = GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])   #optimizer=tf.keras.optimizers.Adam(lr=1e-3

    batch_size = 128

    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        cp_callback
    ]

    model.fit(x_train, y_train, validation_split=0.2, batch_size=batch_size,
              epochs=1, callbacks=callbacks, verbose=1)

    latest = tf.train.latest_checkpoint(checkpoint_dir)

    model.load_weights(latest)

    tokenizer.fit_on_texts(x_test)
    x_test = tokenizer.texts_to_sequences(x_test)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

    score = model.evaluate(x_test, y_test, verbose=1)

    print("Test Score:", score[0])
    print("Test Accuracy:", score[1])

    # Compute the privacy budget expended.
    if FLAGS.dpsgd:
        eps = compute_epsilon(FLAGS.epochs * 60000 // FLAGS.batch_size)
        print('For delta=1e-5, the current epsilon is: %.2f' % eps)
    else:
        print('Trained with vanilla non-private SGD optimizer')

if __name__ == '__main__':
  app.run(main)
