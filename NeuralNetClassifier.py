import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
import NeuralNetHelperFunctions as hf
import NeuralNetModel as nnmodel


print '\nACTIVITY/SPORT CLASSIFICATION USING SENSOR DATA'


################################################################
## RUN PARAMETERS
verbose = False # to display iteration logs
show_val_acc = True # to show average validation accuracies in hyperparameter searches
freqDims = 150
timeDims = 13
channels = 6
num_classes = 5
num_pca_component = 200
pca_whiten = True
hidden_sizes = [1024, 1024]


################################################################
## LOAD DATA
print '\nLoading data...'

data = hf.load_data('../Data/featuresFinal.csv', '../Data/labelsFinal.csv')

train_features = data['train_features'].reshape((-1,freqDims*timeDims,channels))
train_labels   = data['train_labels']
test_features  = data['test_features'].reshape((-1,freqDims*timeDims,channels))
test_labels    = data['test_labels']

print 'Data loaded!'
print 'Training set:  ', train_features.shape[0]
print 'Test set:      ', test_features.shape[0]


################################################################
# REDUCING DIMENSIONALITY USING PCA (separately for each channel)
print '\nReducing dimensionality using PCA...'

reduced_train_features = np.zeros((train_features.shape[0],num_pca_component,channels))
reduced_test_features = np.zeros((test_features.shape[0],num_pca_component,channels))

for c in range(channels):
    pca = PCA(n_components=num_pca_component, whiten=pca_whiten)
    reduced_train_features[:,:,c] = pca.fit_transform(train_features[:,:,c])
    reduced_test_features[:,:,c]  = pca.transform(test_features[:,:,c])

print 'Dimensionality reduced!'
print 'Old feature dimensions:', train_features.shape[1]*train_features.shape[2]
print 'New feature dimensions:', reduced_train_features.shape[1]*reduced_train_features.shape[2]


################################################################
## HYPERPARAMETER OPTIMIZATION
print '\nOptimizing hyperparameters...'

best_val_acc = None
best_lr = None
best_keep_prob = None

lrs = np.logspace(-3,0,10)
keep_probs = np.linspace(0.1,1,10)
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

            model = nnmodel.NeuralNetModel(sess=sess, num_components=num_pca_component,
                                           channels=channels, num_classes=num_classes, hidden_sizes=hidden_sizes)
            model.train(train_fold_features, train_fold_labels,
                        learning_rate=lr, keep_prob=keep_prob,
                        batch_size=25, num_epochs=10, verbose=verbose)

            val_predictions = model.predict(val_fold_features)
            avg_val_acc += model.check_accuracy(val_predictions, np.argmax(val_fold_labels, 1))

            # close tensorflow session
            sess.close()

        avg_val_acc /= num_folds

        if verbose or show_val_acc:
            print 'Learning rate: %.3e    Dropout keep probability: %.1f    Average validation accuracy: %.4f' % (lr, keep_prob, avg_val_acc)

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_lr = lr
            best_keep_prob = keep_prob

print 'Hyperparameters optimized!'
print 'Optimal learning rate: %.3e' % best_lr
print 'Optimal dropout keep probability: %.1f' % best_keep_prob


################################################################
## TRAIN A MODEL WITH OPTIMAL HYPERPARAMETERS AND TEST IT
print '\nRetraining model with optimal hyperparameters...'

# start tensorflow session
tf.reset_default_graph()
sess = tf.Session()

model = nnmodel.NeuralNetModel(sess=sess, num_components=num_pca_component,
                               channels=channels, num_classes=num_classes, hidden_sizes=hidden_sizes)
model.train(reduced_train_features, train_labels,
            learning_rate=best_lr, keep_prob=best_keep_prob,
            batch_size=25, num_epochs=40, verbose=verbose)

print 'Model trained!'

test_predictions = model.predict(reduced_test_features)
test_acc = model.check_accuracy(test_predictions, np.argmax(test_labels, 1))
print 'Test accuracy: %.4f' % test_acc

# close tensorflow session
sess.close()


################################################################
## ANALYZE FAILED CASES
sports = ['Badminton', 'Basketball', 'Running', 'Skating', 'Walking']

wrongly_predicated_test_labels = test_labels[test_predictions != np.argmax(test_labels, 1)]
sportwise_errors = np.zeros(test_labels.shape[1], dtype='int')
sportwise_examples = np.zeros(test_labels.shape[1], dtype='int')

print '\nTEST ERRORS: (# of errors / # of test examples)'

for i in range(test_labels.shape[1]):
    sportwise_errors[i] = np.sum(np.equal(wrongly_predicated_test_labels[:,i], np.ones(wrongly_predicated_test_labels.shape[0])))
    sportwise_examples[i] = np.sum(np.equal(test_labels[:,i], np.ones(test_labels.shape[0])))
    print sports[i]+':', sportwise_errors[i], '/', sportwise_examples[i]
