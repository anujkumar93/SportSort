import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import tensorflow as tf
import sys


def weight_var(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_var(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv(x, W):
    # 2D convolution
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x):
    # 2x2 max pooling
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def sample_fractions(labels):
    N, C = labels.shape
    fractions = np.zeros(C)
    if N==0:
        return fractions

    for i in range(C):
        fractions[i] = np.sum(np.equal(labels[:, i], np.ones(N)))
    fractions /= N

    return fractions

def load_data(features_csv, labels_csv, newPerson=False):
    # loads data as a dictionary of numpy arrays
    data = {}

    data_split = np.array([80.0,20.0]) # split sizes for train and test sets
    if newPerson:
        data_split = np.array([0.0, 100.0])

    features = np.loadtxt(features_csv, dtype='float', delimiter=', ')
    labels = np.loadtxt(labels_csv, dtype='int', delimiter=', ')

    assert features.shape[0] == labels.shape[0] # sanity check

    N, D = features.shape
    _, C = labels.shape

    index_test_start = np.floor((data_split[0] / np.sum(data_split)) * N).astype(int)

    fractions_overall = sample_fractions(labels)

    shuffled_order = None
    shuffled_labels = None
    while True:
        shuffled_order = np.random.permutation(N)
        shuffled_labels = labels[shuffled_order]

        if newPerson:
            break
        else:
            train_fractions = sample_fractions(shuffled_labels[:index_test_start])
            test_fractions = sample_fractions(shuffled_labels[index_test_start:])

            ssd_train = np.sum((train_fractions - fractions_overall) ** 2)
            ssd_test = np.sum((test_fractions - fractions_overall) ** 2)
            if ssd_train<0.0005 and ssd_test<0.0005:
                break

    assert shuffled_order is not None
    assert shuffled_labels is not None

    shuffled_features = features[shuffled_order]

    data['train_features'] = shuffled_features[:index_test_start]
    data['train_labels']   = shuffled_labels[:index_test_start]
    data['test_features']  = shuffled_features[index_test_start:]
    data['test_labels']    = shuffled_labels[index_test_start:]

    return data

def get_data_folds(k, num_folds, features, labels):
    # returns k-th fold for training and validation data split
    N = features.shape[0]
    samples_per_fold = int(np.ceil(float(N) / num_folds))

    start_index = k*samples_per_fold
    end_index = (k+1)*samples_per_fold
    if end_index > N:
        end_index = N

    train_fold_features = np.delete(features, np.arange(start_index, end_index), axis=0)
    train_fold_labels   = np.delete(labels, np.arange(start_index, end_index), axis=0)
    val_fold_features   = features[start_index:end_index]
    val_fold_labels     = labels[start_index:end_index]

    return train_fold_features, train_fold_labels, val_fold_features, val_fold_labels

class Logger(object):
    """
    Logger class to print stdout messages into a log file while displaying them in stdout also.
    Pass filename to which to save when instantiating.
    """
    def __init__(self, f_log_name):
        """
        Initialization.
        :param f_log_name: file path to which to write the logs
        """
        self.terminal = sys.stdout
        self.log = open(f_log_name, 'w')

    def write(self, message):
        """
        Actual logging operations.
        :param message: The message to log.
        :return: Nothing
        """
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        """
        Exists only to satisfy Python 3
        :return: Nothing
        """
        pass

    def close_log(self):
        """
        Closes the opened log file.
        :return:
        """
        self.log.close()

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exists to close the log file in case user terminated the script, etc., and close_log is not explicitly called.
        """
        self.close_log()
