import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import DataPreprocessing as dp
import FeatureExtraction as fe
import NeuralNetClassifier as nnc
import RandomForestClassifier as rfc
import ClassifierHelperFunctions as hf
import sys
import os
import time
import datetime


print '\nACTIVITY/SPORT CLASSIFICATION USING SENSOR DATA'


# Log parameters
verbose = False
show_val_acc = True

# Fixed parameters
sports = ['Badminton','Basketball','Foosball','Running','Skating','Walking']
channels = 6
pca_whiten = False
trimLength = 15
hidden_sizes = [1024,1024]

# Variable parameters
numSecondsPerImage_options = [15, 30]
fftWidth_options = [1, 2, 6]
fftJump_options = [1, 2]
num_pca_components_options = [200, 400]
algo_switch_options = [1, 2, 3]


################################################################
# !!! SHOULD RESET THIS AT THE END !!!
# Set up printing out a log (redirects all prints to the file)
orig_stdout = sys.stdout
f_log_name = '../Results/' + os.path.basename(__file__) \
             + '_' \
             + datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S') \
             + '.log'
logger = hf.Logger(f_log_name)
sys.stdout = logger

for algoSwitch in algo_switch_options:
    
    for numSecondsPerImage in numSecondsPerImage_options:
        
        dp.data_preprocessing(sports=sports, secondsToKeep=numSecondsPerImage, trimLength=trimLength,\
                              switchAlgo=algoSwitch)
        if algoSwitch!=3:
            dp.data_preprocessing(sports=sports, secondsToKeep=numSecondsPerImage, trimLength=trimLength,\
                              switchAlgo=algoSwitch, newPerson=True)

        for fftWidth in fftWidth_options:
            for fftJump in fftJump_options:

                fe.feature_extraction(sports=sports, numSecondsPerImage=numSecondsPerImage, fftWidth=fftWidth,
                                      fftJump=fftJump, switchAlgo=algoSwitch)
                if algoSwitch!=3:
                    fe.feature_extraction(sports=sports, numSecondsPerImage=numSecondsPerImage, fftWidth=fftWidth,
                                      fftJump=fftJump, newPerson=True, switchAlgo=algoSwitch)

                for num_pca_components in num_pca_components_options:

                    # Derived parameters
                    freqDims = 50 * fftWidth / 2
                    timeDims = (numSecondsPerImage - fftWidth) / fftJump + 1

                    # Log current run parameters
                    print 'sports =', sports
                    print 'trimLength =', trimLength
                    print 'numSecondsPerImage =', numSecondsPerImage
                    print 'fftWidth =', fftWidth
                    print 'fftJump =', fftJump
                    print 'channels =', channels
                    print 'num_pca_components =', num_pca_components
                    print 'pca_whiten =', pca_whiten
                    print 'hidden_sizes =', hidden_sizes
                    print 'verbose =', verbose
                    print 'show_val_acc =', show_val_acc
                    print 'freqDims =', freqDims
                    print 'timeDims =', timeDims

                    # CLASSIFIERS

                    nnc.run_neural_net_classifier(sports=sports, freqDims=freqDims, timeDims=timeDims,
                                                  channels=channels, num_pca_components=num_pca_components,
                                                  pca_whiten=pca_whiten, hidden_sizes=hidden_sizes, verbose=verbose,
                                                  show_val_acc=show_val_acc)

                    rfc.run_random_forest_classifier(sports=sports, featuresFile='../Data/featuresFinal.csv',
                                                     labelsFile='../Data/labelsFinal.csv', freqDims=freqDims,
                                                     timeDims=timeDims, channels=channels,
                                                     num_pca_components=num_pca_components, pca_whiten=pca_whiten)


################################################################
# !!! RESETTING STDOUT LOGGING !!!
# Stop redirecting pring out to log
logger.close_log()
sys.stdout = orig_stdout