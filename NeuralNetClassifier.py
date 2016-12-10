import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
import ClassifierHelperFunctions as hf
import NeuralNetModel as nnmodel


def run_neural_net_classifier(sports=['Badminton', 'Basketball', 'Foosball', 'Running', 'Skating', 'Walking'],
                              freqDims=150,
                              timeDims=13,
                              channels=6,
                              num_pca_components=200,
                              pca_whiten=True,
                              hidden_sizes=[1024, 1024],
                              verbose=False,
                              show_val_acc=True):
    """
    Loads data, and runs a neural network classifier to classify activities.
    :param sports: List of names of activities in data set
    :param freqDims: Number of steps in frequency dimension in input feature data
    :param timeDims: Number of steps in time dimension in input feature data
    :param channels: Number of channels in input feature data
    :param num_pca_components: Number of components wanted after PCA
    :param pca_whiten: Set True to whiten the PCA data
    :param hidden_sizes: list of sizes for the hidden layers
    :param verbose: Set True to display iteration logs
    :param show_val_acc: Set True to show average validation accuracies in hyperparameter searches
    :return: Nothing
    """

    num_classes = len(sports)

    ################################################################
    ## LOAD DATA
    print '\nLoading data...'

    data = hf.load_data('../Data/featuresFinal.csv', '../Data/labelsFinal.csv')
    train_features = data['train_features'].reshape((-1,freqDims*timeDims,channels))
    train_labels   = data['train_labels']
    test_features  = data['test_features'].reshape((-1,freqDims*timeDims,channels))
    test_labels    = data['test_labels']

    newPerson_data = hf.load_data('../Data/newPersonFeaturesFinal.csv', '../Data/newPersonLabelsFinal.csv', newPerson=True)
    newPerson_test_features = newPerson_data['test_features'].reshape((-1,freqDims*timeDims,channels))
    newPerson_test_labels   = newPerson_data['test_labels']

    print 'Data loaded!'
    print 'Training set:       ', train_features.shape[0]
    print 'Test set:           ', test_features.shape[0]
    print 'New person test set:', newPerson_test_features.shape[0]


    ################################################################
    # REDUCING DIMENSIONALITY USING PCA (separately for each channel)
    print '\nReducing dimensionality using PCA...'

    reduced_train_features = np.zeros((train_features.shape[0], num_pca_components, channels))
    reduced_test_features = np.zeros((test_features.shape[0], num_pca_components, channels))
    reduced_newPerson_test_features = np.zeros((newPerson_test_features.shape[0], num_pca_components, channels))

    for c in range(channels):
        pca = PCA(n_components=num_pca_components, whiten=pca_whiten)
        reduced_train_features[:,:,c] = pca.fit_transform(train_features[:,:,c])
        reduced_test_features[:,:,c]  = pca.transform(test_features[:,:,c])
        reduced_newPerson_test_features[:, :, c] = pca.transform(newPerson_test_features[:, :, c])

    print 'Dimensionality reduced!'
    print 'Old feature dimensions:', train_features.shape[1]*train_features.shape[2]
    print 'New feature dimensions:', reduced_train_features.shape[1]*reduced_train_features.shape[2]


    ################################################################
    ## HYPERPARAMETER OPTIMIZATION
    print '\nOptimizing hyperparameters...\n'

    best_val_acc = None
    best_lr = None
    best_keep_prob = None

    lrs = np.logspace(-4,1,11)
    keep_probs = np.linspace(0.2,1,5)
    num_folds = 4

    for lr in lrs:
        for keep_prob in keep_probs:
            avg_val_acc = 0.0

            for k in range(num_folds):
                # get training and validations sets for this fold
                train_fold_features, train_fold_labels, val_fold_features, val_fold_labels = \
                        hf.get_data_folds(k, num_folds, reduced_train_features, train_labels)

                # start tensorflow session
                tf.reset_default_graph()
                sess = tf.Session()

                model = nnmodel.NeuralNetModel(sess=sess, num_components=num_pca_components,
                                               channels=channels, num_classes=num_classes, hidden_sizes=hidden_sizes)
                model.train(train_fold_features, train_fold_labels,
                            learning_rate=lr, keep_prob=keep_prob,
                            batch_size=50, num_epochs=20, verbose=verbose)

                val_predictions = model.predict(val_fold_features)
                avg_val_acc += model.check_accuracy(val_predictions, np.argmax(val_fold_labels, 1))

                # close tensorflow session
                sess.close()

            avg_val_acc /= num_folds

            if verbose or show_val_acc:
                print 'ls: %.3e\t\tkeep_prob: %.1f\t\tavg_val_acc: %.4f' % (lr, keep_prob, avg_val_acc)

            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                best_lr = lr
                best_keep_prob = keep_prob

    print '\nHyperparameters optimized!'
    print 'Optimal learning rate: %.3e' % best_lr
    print 'Optimal dropout keep probability: %.1f' % best_keep_prob


    ################################################################
    ## TRAIN A MODEL WITH OPTIMAL HYPERPARAMETERS AND TEST IT
    print '\nRetraining model with optimal hyperparameters...'

    # start tensorflow session
    tf.reset_default_graph()
    sess = tf.Session()

    model = nnmodel.NeuralNetModel(sess=sess, num_components=num_pca_components,
                                   channels=channels, num_classes=num_classes, hidden_sizes=hidden_sizes)
    model.train(reduced_train_features, train_labels,
                learning_rate=best_lr, keep_prob=best_keep_prob,
                batch_size=50, num_epochs=40, verbose=verbose)

    print 'Model trained!'

    print '\n\nACCURACY'

    train_predictions = model.predict(reduced_train_features)
    train_acc = model.check_accuracy(train_predictions, np.argmax(train_labels, 1))
    print 'On training set:        %.4f' % train_acc

    test_predictions = model.predict(reduced_test_features)
    test_acc = model.check_accuracy(test_predictions, np.argmax(test_labels, 1))
    print 'On test set:            %.4f' % test_acc

    newPerson_test_predictions = model.predict(reduced_newPerson_test_features)
    newPerson_test_acc = model.check_accuracy(newPerson_test_predictions, np.argmax(newPerson_test_labels, 1))
    print 'On new person test set: %.4f' % newPerson_test_acc

    # close tensorflow session
    sess.close()


    ################################################################
    ## ANALYZE FAILED CASES
    print '\n\nTEST ERRORS (# of errors / # of test examples)'

    predicted_vs_truth_count_matrix = np.zeros([len(sports), len(sports)], dtype='int')
    for i in range(test_predictions.shape[0]):
        predicted_vs_truth_count_matrix[np.argmax(test_labels, 1)[i], test_predictions[i]] += 1

    for i in range(len(sports)):
        num_correctly_predicted = predicted_vs_truth_count_matrix[i, i]
        num_actually_were       = np.sum(predicted_vs_truth_count_matrix[i, :])
        num_wrongly_predicted   = num_actually_were - num_correctly_predicted
        print sports[i] + ':\t', num_wrongly_predicted, '/', num_actually_were, '\t', predicted_vs_truth_count_matrix[i,:]

    print '\n\nTEST ERRORS FOR NEW PERSON (# of errors / # of test examples)'

    newPerson_predicted_vs_truth_count_matrix = np.zeros([len(sports), len(sports)], dtype='int')
    for i in range(newPerson_test_predictions.shape[0]):
        newPerson_predicted_vs_truth_count_matrix[np.argmax(newPerson_test_labels, 1)[i], newPerson_test_predictions[i]] += 1

    for i in range(len(sports)):
        num_correctly_predicted = newPerson_predicted_vs_truth_count_matrix[i, i]
        num_actually_were = np.sum(newPerson_predicted_vs_truth_count_matrix[i, :])
        num_wrongly_predicted = num_actually_were - num_correctly_predicted
        print sports[i] + ':\t', num_wrongly_predicted, '/', num_actually_were, '\t', newPerson_predicted_vs_truth_count_matrix[i, :]
