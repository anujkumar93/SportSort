import DataPreprocessing as dp
import FeatureExtraction as fe
import NeuralNetClassifier as nnc
import RandomForestClassifier as rfc


print '\nACTIVITY/SPORT CLASSIFICATION USING SENSOR DATA'


# Run parameters
sports = ['Badminton','Basketball','Foosball','Running','Skating','Walking']
trimLength = 15
numSecondsPerImage = 30
fftWidth = 6
fftJump = 2
channels = 6
num_pca_components = 200
pca_whiten = True
hidden_sizes = [1024, 1024]
verbose = False
show_val_acc = True

# Derived parameters
freqDims = 50 * fftWidth / 2
timeDims = (numSecondsPerImage - fftWidth) / fftJump + 1

# Run methods
dp.data_preprocessing(sports=sports, secondsToKeep=numSecondsPerImage, trimLength=trimLength)
fe.feature_extraction(sports=sports, numSecondsPerImage=numSecondsPerImage, fftWidth=fftWidth, fftJump=fftJump)
nnc.run_neural_net_classifier(sports=sports, freqDims=freqDims, timeDims=timeDims, channels=channels,
                              num_pca_components=num_pca_components, pca_whiten=pca_whiten, hidden_sizes=hidden_sizes,
                              verbose=verbose, show_val_acc=show_val_acc)
rfc.run_random_forest_classifier(sports=sports,
                                 featuresFile='../Data/featuresFinal.csv',
                                 labelsFile='../Data/labelsFinal.csv',
                                 freqDims=freqDims, timeDims=timeDims, channels=channels,
                                 num_pca_component=200,
                                 pca_whiten=True)
