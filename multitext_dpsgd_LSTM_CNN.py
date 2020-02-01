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
from sklearn import metrics
from sklearn.metrics import log_loss, accuracy_score
from sklearn.metrics import confusion_matrix


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


# vocab_size = 5000
embedding_dim = 100
max_length = 200


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
    print(len(labels))
    print(len(articles))

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(articles)

    word_index = tokenizer.word_index
    vocab_size = len(word_index)

    sequences = tokenizer.texts_to_sequences(articles)
    padded = pad_sequences(sequences, maxlen=max_length)

    train_size = int(len(articles) * 0.7)
    validation_size = int(len(articles) * 0.2)

    training_sequences = padded[0:train_size]
    train_labels = labels[0:train_size]

    validation_sequences = padded[train_size:train_size + validation_size]
    validation_labels = labels[train_size:train_size + validation_size]

    test_sequences = padded[train_size + validation_size:]
    test_labels = labels[train_size + validation_size:]

    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)

    training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
    validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))
    test_label_seq = np.array(label_tokenizer.texts_to_sequences(test_labels))

    print(training_sequences.shape)
    print(validation_sequences.shape)
    print(training_label_seq.shape)
    print(validation_label_seq.shape)

    print(vocab_size)
    print(word_index['i'])

    embeddings_index = {};
    with open('embedding/glove.6B/glove.6B.100d.txt') as f:
        for line in f:
            values = line.split();
            word = values[0];
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs;

    embeddings_matrix = np.zeros((vocab_size + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector

    print(len(embeddings_matrix))

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size + 1, embedding_dim, input_length=max_length, weights=[embeddings_matrix],
                                  trainable=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(64, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=4),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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
    history = model.fit(training_sequences, training_label_seq, epochs=num_epochs,
                        validation_data=(validation_sequences, validation_label_seq), verbose=2)

    plot_graphs(history, "accuracy")
    plot_graphs(history, "loss")


    scores = model.evaluate(test_sequences, test_label_seq, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    output_test = model.predict(test_sequences)
    print(np.shape(output_test))
    final_pred = np.argmax(output_test, axis=1)
    print(np.shape(final_pred))

    final_pred_list = np.reshape(final_pred, (len(test_sequences), 1))

    results = confusion_matrix(test_label_seq, final_pred_list)
    print("Confusion Matrix =" , results)

    print(np.shape(test_label_seq))
    print(np.shape(final_pred_list))

    precisions, recall, f1_score, true_sum = metrics.precision_recall_fscore_support(test_label_seq, final_pred_list)
    print("Multi-label Classification LSTM CNN Precision =", precisions)
    print("Multi-label Classification LSTM CNN Recall=", recall)
    print("Multi-label Classification LSTM CNN F1 Score =", f1_score)
    print('Multi-label Classification Accuracy: {}'.format((accuracy_score(test_label_seq, final_pred_list))))

    # print(test_label_seq)
    # print(final_pred_list)

    classes = np.array(range(0, 6))
    # print('Log loss: {}'.format(log_loss(classes[np.argmax(test_label_seq, axis=1)], output_test)))

    # Compute the privacy budget expended.
    if FLAGS.dpsgd:
        eps = compute_epsilon(FLAGS.epochs * 60000 // FLAGS.batch_size)
        print('For delta=1e-5, the current epsilon is: %.2f' % eps)
    else:
        print('Trained with vanilla non-private SGD optimizer')


if __name__ == '__main__':
  app.run(main)
