import numpy as np
import tensorflow as tf
import NeuralNetHelperFunctions as hf

class NeuralNetModel:
    def __init__(self, sess, num_components=100, channels=6, num_classes=5, hidden_sizes=[1024, 1024]):
        self.sess = sess

        # placeholders for input data and hyperparameters
        self.x = tf.placeholder(tf.float32, shape=[None, num_components, channels])
        self.y = tf.placeholder(tf.float32, shape=[None, num_classes])
        self.keep_prob = tf.placeholder(tf.float32)
        self.learning_rate = tf.placeholder(tf.float32)

        self.reshaped_x = tf.reshape(self.x, [-1, num_components * channels])

        self.W_fc1 = hf.weight_var([num_components * channels, hidden_sizes[0]])
        self.b_fc1 = hf.bias_var([hidden_sizes[0]])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.reshaped_x, self.W_fc1) + self.b_fc1)

        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

        self.W_fc2 = hf.weight_var([hidden_sizes[0], hidden_sizes[1]])
        self.b_fc2 = hf.bias_var([hidden_sizes[1]])
        self.h_fc2 = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2

        self.h_fc2_drop = tf.nn.dropout(self.h_fc2, self.keep_prob)

        self.W_fc3 = hf.weight_var([hidden_sizes[1], num_classes])
        self.b_fc3 = hf.bias_var([num_classes])
        self.y_out = tf.matmul(self.h_fc2_drop, self.W_fc3) + self.b_fc3

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y_out, self.y))
        self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cross_entropy)
        self.predictions = tf.argmax(self.y_out, 1)
        self.correct_prediction = tf.equal(self.predictions, tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.sess.run(tf.global_variables_initializer())

    def train(self, features, labels, learning_rate=1e-2, keep_prob=0.5, batch_size=25, num_epochs=20, verbose=True):
        # tensorflow neural net training iterations
        iters_per_epoch = np.ceil(float(features.shape[0]) / batch_size).astype(int)

        for i in range(num_epochs * iters_per_epoch):
            start_index = (i%iters_per_epoch)*batch_size
            end_index = ((i%iters_per_epoch)+1)*batch_size
            if end_index > features.shape[0]:
                end_index = features.shape[0]

            features_batch = features[start_index:end_index]
            labels_batch = labels[start_index:end_index]

            if i % 50 == 0 and verbose:
                with self.sess.as_default():
                    train_accuracy = self.accuracy.eval(feed_dict={self.x: features_batch, self.y: labels_batch,
                                                                   self.keep_prob: 1.0})
                    print("step %d, training accuracy: %.4f" % (i, train_accuracy))

            with self.sess.as_default():
                self.train_step.run(feed_dict={self.x: features_batch, self.y: labels_batch,
                                               self.keep_prob: keep_prob, self.learning_rate: learning_rate})

    def predict(self, features, batch_size=50):
        # return predicted labels for given input features
        y_pred = np.array([])

        for i in range(np.ceil(float(features.shape[0]) / batch_size).astype(int)):
            start_index = i * batch_size
            end_index = (i + 1) * batch_size
            if end_index > features.shape[0]:
                end_index = features.shape[0]

            features_batch = features[start_index:end_index]
            with self.sess.as_default():
                y_pred = np.append(y_pred, self.predictions.eval(feed_dict={self.x: features_batch, self.keep_prob: 1.0}))

        return y_pred.astype(int)

    @staticmethod
    def check_accuracy(predicted_labels, true_labels, batch_size=50):
        # calculates accuracy in batches to save memory consumption
        assert predicted_labels.shape == true_labels.shape

        y_accs = np.array([])

        for i in range(np.ceil(float(predicted_labels.shape[0]) / batch_size).astype(int)):
            start_index = i*batch_size
            end_index = (i+1)*batch_size
            if end_index > predicted_labels.shape[0]:
                end_index = predicted_labels.shape[0]

            y_accs = np.append(y_accs, np.equal(predicted_labels[start_index:end_index], true_labels[start_index:end_index]))

        return np.mean(y_accs)
