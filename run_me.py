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
pca_whiten = True
trimLength = 15
hidden_sizes = [1024,1024]

# Variable parameters
numSecondsPerImage_options = [15, 30]
fftWidth_options = [4, 6]
fftJump_options = [1]
num_pca_components_options = [100, 200]
algo_switch_options = [0, 1, 2, 3]


for algoSwitch in algo_switch_options:

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

    for numSecondsPerImage in numSecondsPerImage_options:

        if algoSwitch <= 1:
            dp.data_preprocessing(sports=sports, secondsToKeep=numSecondsPerImage, trimLength=trimLength,
                              switchAlgo=algoSwitch)
        if algoSwitch == 2:
            dp.data_preprocessing(sports=sports, secondsToKeep=numSecondsPerImage, trimLength=trimLength,
                                  switchAlgo=0)
            dp.data_preprocessing(sports=sports, secondsToKeep=numSecondsPerImage, trimLength=trimLength,
                                  switchAlgo=algoSwitch)
        if algoSwitch == 3:
            dp.data_preprocessing(sports=sports, secondsToKeep=numSecondsPerImage, trimLength=trimLength,
                                  switchAlgo=4)
            dp.data_preprocessing(sports=sports, secondsToKeep=numSecondsPerImage, trimLength=trimLength,
                                  switchAlgo=algoSwitch)

        for fftWidth in fftWidth_options:
            for fftJump in fftJump_options:

                fe.feature_extraction(sports=sports, numSecondsPerImage=numSecondsPerImage, fftWidth=fftWidth,
                                      fftJump=fftJump, finalOrNew=0)
                if algoSwitch > 1:
                    fe.feature_extraction(sports=sports, numSecondsPerImage=numSecondsPerImage, fftWidth=fftWidth,
                                          fftJump=fftJump, finalOrNew=1)

                for num_pca_components in num_pca_components_options:

                    # Derived parameters
                    freqDims = 50 * fftWidth / 2
                    timeDims = (numSecondsPerImage - fftWidth) / fftJump + 1

                    # Log current run parameters
                    print 'algoSwitch =', algoSwitch
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
                                                  show_val_acc=show_val_acc, algoSwitch=algoSwitch)

                    rfc.run_random_forest_classifier(sports=sports, featuresFile='../Data/featuresFinal.csv',
                                                     labelsFile='../Data/labelsFinal.csv', freqDims=freqDims,
                                                     timeDims=timeDims, channels=channels,
                                                     num_pca_components=num_pca_components, pca_whiten=pca_whiten,
                                                     algoSwitch=algoSwitch)


    ################################################################
    # !!! RESETTING STDOUT LOGGING !!!
    # Stop redirecting pring out to log
    logger.close_log()
    sys.stdout = orig_stdout
