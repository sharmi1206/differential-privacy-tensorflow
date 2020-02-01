import tensorflow as tf
print(tf.__version__)
import matplotlib.pyplot as plt

from absl import app
from absl import flags
from absl import logging
import os


import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPAdamGaussianOptimizer


GradientDescentOptimizer = tf.compat.v1.train.GradientDescentOptimizer
AdamOptimizer = tf.compat.v1.train.AdamOptimizer

flags.DEFINE_boolean(
    'dpsgd', True, 'If True, train with DP-SGD. If False, '
    'train with vanilla SGD.')
flags.DEFINE_float('learning_rate', 0.005, 'Learning rate for training')
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


vocab_size = 5000
embedding_dim = 64
max_length = 200
training_portion = .8


def decode_article(text, reverse_word_index):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()



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

    articles = []
    labels = []

    with open("data/bbc-text.csv", 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            labels.append(row[0])
            article = row[1]
            for word in STOPWORDS:
                token = ' ' + word + ' '
                article = article.replace(token, ' ')
                article = article.replace(' ', ' ')
            articles.append(article)



    train_size = int(len(articles) * training_portion)
    train_articles = articles[0: train_size]
    train_labels = labels[0: train_size]

    validation_articles = articles[train_size:]
    validation_labels = labels[train_size:]

    tokenizer = Tokenizer(num_words = vocab_size)
    tokenizer.fit_on_texts(train_articles)
    word_index = tokenizer.word_index


    dict(list(word_index.items())[0:10])

    train_sequences = tokenizer.texts_to_sequences(train_articles)
    print(train_sequences[10])

    train_padded = pad_sequences(train_sequences, maxlen=max_length)

    print(len(train_sequences[0]))
    print(len(train_padded[0]))

    print(len(train_sequences[1]))
    print(len(train_padded[1]))

    print(len(train_sequences[10]))
    print(len(train_padded[10]))

    print(train_sequences[10])
    print(train_padded[10])
    print(train_sequences[0])

    print(train_padded[0])

    validation_sequences = tokenizer.texts_to_sequences(validation_articles)
    validation_padded = pad_sequences(validation_sequences, maxlen=max_length)

    print(len(validation_sequences))
    print(validation_padded.shape)

    print(set(labels))

    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)

    training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
    validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))


    print(training_label_seq[0])
    print(training_label_seq[1])
    print(training_label_seq[2])
    print(training_label_seq.shape)

    print(validation_label_seq[0])
    print(validation_label_seq[1])
    print(validation_label_seq[2])
    print(validation_label_seq.shape)

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


    print(decode_article(train_padded[10], reverse_word_index))
    print('---')
    print(train_articles[10])


    model = tf.keras.Sequential([
        # Add an Embedding layer expecting input vocab of size 5000, and output embedding dimension of size 64 we set at the top
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
    #    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        # use ReLU in place of tanh function since they are very good alternatives of each other.
        tf.keras.layers.Dense(embedding_dim, activation='relu'),
        # Add a Dense layer with 6 units and softmax activation.
        # When we have multiple outputs, softmax convert outputs layers into a probability distribution.
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    model.summary()
    if FLAGS.dpsgd:
        optimizer = DPAdamGaussianOptimizer(
            l2_norm_clip=FLAGS.l2_norm_clip,
            noise_multiplier=FLAGS.noise_multiplier,
            num_microbatches=FLAGS.microbatches,
            learning_rate=FLAGS.learning_rate)
          # reduction=tf.compat.v1.losses.Reduction.NONE
    else:
        optimizer = AdamOptimizer()


    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    num_epochs = 10
    history = model.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=2)

    plot_graphs(history, "accuracy")
    plot_graphs(history, "loss")


    txt = ["A WeWork shareholder has taken the company to court over the near-$1.7bn (£1.3bn) leaving package approved for ousted co-founder Adam Neumann."]
    seq = tokenizer.texts_to_sequences(txt)
    padded = pad_sequences(seq, maxlen=max_length)
    pred = model.predict(padded)
    labels = ['sport', 'bussiness', 'politics', 'tech', 'entertainment']
    print(pred, labels[np.argmax(pred)-1])

    # Compute the privacy budget expended.
    if FLAGS.dpsgd:
        eps = compute_epsilon(FLAGS.epochs * 60000 // FLAGS.batch_size)
        print('For delta=1e-5, the current epsilon is: %.2f' % eps)
    else:
        print('Trained with vanilla non-private SGD optimizer')


if __name__ == '__main__':
  app.run(main)
