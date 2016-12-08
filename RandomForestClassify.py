import numpy as np
from sklearn.ensemble import  RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV

def sample_fractions(labels):
    N, C = labels.shape
    fractions = np.zeros(C)
    for i in range(C):
        fractions[i] = np.sum(np.equal(labels[:, i], np.ones(N)))
    fractions /= N
    return fractions

def load_data(features_csv, labels_csv):
    # loads data as a dictionary of numpy arrays
    data = {}

    data_split = np.array([80.0,20.0]) # split sizes for train and test sets

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

freqDim=150
timeDims = 13
channels = 6
num_classes = 6
num_pca_component=200
pca_whiten=True

print '\nLoading data...'

data = load_data('./Data/featuresFinal.csv', './Data/labelsFinal.csv')

train_features = data['train_features'].reshape((-1,freqDim*timeDims,channels))
train_labels   = data['train_labels']
test_features  = data['test_features'].reshape((-1,freqDim*timeDims,channels))
test_labels    = data['test_labels']

print 'Data loaded!'
print 'Training set:  ', train_features.shape
print 'Test set:      ', test_features.shape

random_forest=RandomForestClassifier()
tuned_parameters = [
#                     {'pca__n_components' : [100,200,400], 'classifier__min_samples_split' : [5,10], 
#                      'classifier__criterion' : ['gini','entropy'], 'classifier__n_estimators' : [15,25]
#                      }
                   {'pca' : [None], 'classifier__min_samples_split' : [2,5,10], 
                     'classifier__criterion' : ['gini','entropy'], 'classifier__n_estimators' : [15,25,30]
                     }
                    ] #prepared the range of parameters to search over for GridSearchCV


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

pipeline = Pipeline([ ("pca",pca),("classifier", random_forest)])
pipeline_grid = GridSearchCV(pipeline, tuned_parameters, cv = 5, n_jobs=-1)
train_labels   = np.argmax(train_labels, 1)

reduced_train_features=reduced_train_features.reshape((reduced_train_features.shape[0],-1))
reduced_test_features=reduced_test_features.reshape((reduced_test_features.shape[0],-1))

pipeline_grid.fit(reduced_train_features,train_labels)
print pipeline_grid.best_params_
predictions=pipeline_grid.predict(reduced_test_features)
print predictions.shape

sports = ['Badminton', 'Basketball', 'Foosball', 'Running', 'Skating', 'Walking']

print '\nTEST ERRORS (# of errors / # of test examples)'

predicted_vs_truth_count_matrix = np.zeros([len(sports), len(sports)], dtype='int')
for i in range(predictions.shape[0]):
    predicted_vs_truth_count_matrix[np.argmax(test_labels, 1)[i], predictions[i]] += 1

for i in range(len(sports)):
    num_correctly_predicted = predicted_vs_truth_count_matrix[i,i]
    num_wrongly_predicted   = np.sum(predicted_vs_truth_count_matrix[i,:]) - num_correctly_predicted
    num_actually_were       = np.sum(predicted_vs_truth_count_matrix[i,:])
    print sports[i] + ':\t', num_wrongly_predicted, '/', num_actually_were, '\t', predicted_vs_truth_count_matrix[i,:]