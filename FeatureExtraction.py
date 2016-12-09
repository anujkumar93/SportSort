import numpy as np
import scipy.fftpack
import FeatureExtractionHelperFunctions as fehp


def feature_extraction(sports=['Badminton','Basketball','Foosball','Running','Skating','Walking'],
                       numSecondsPerImage=30,
                       fftWidth=6,
                       fftJump=2):
    """
    Extracts features for the samples which are meant to be fed to a classifier later.
    Final features are:
    - 3-dimensional arrays of FFT features
        - element values: power of frequency
        - 1st dim:        frequency
        - 2nd dim:        time
        - 3rd dim:        sensor channel
    - A set of statistical features
        - 2nd dim:        time
        - 3rd dim:        sensor channel
        - 1st dim:        has 16 rows corresponding to following calculations on raw data
        #     The 16 rows are: 1.mean; 2.standard deviation; 3.coefficient of variation; 4.peak-to-peak amplitude
        #     5-9.10th, 25th, 50th, 75th, 90th percentiles; 10.inter-quartile range; 11.lag-one autocorrelation; 
        #     12.skewedness; 13.kurtosis; 14.signal power; 15.log-energy; 16.zero-crossings
        Once this matrix is converted to a 2D array, we add another set of features to this, which correlates each
        of the 6 channels, further producing 15 new feature columns. 

    :param sports:
    :param numSecondsPerImage:
    :param fftWidth:
    :param fftJump:
    :return: 3 outputFiles containing FFT features data, statistical features data, and labels data
    """

    numColumns = (numSecondsPerImage - fftWidth) / fftJump + 1  # total no of columns to include in one image

    outputLabels = None
    outputArray  = None
    outputSecondaryArray = None

    for ind in range(len(sports)):
        print '\n'+sports[ind]

        fileName = '../Data/' + sports[ind] + '/Final.csv'

        data = np.loadtxt(fileName, dtype='string', delimiter=', ')
        counter = 1
        firstSensor = "Accelerometer"
        secondSensor = "Gyroscope"

        totalNumOfLines = fehp.file_len(fileName)

        with open(fileName, 'r') as fileStream:
            firstLine = fileStream.readline()
            firstWord = firstLine.split(', ')[0]
            if "gyro" in firstWord.lower():
                firstSensor = "Gyroscope"
            while True:
                line = fileStream.readline()
                currFirstWord = line.split(', ')[0]
                if "gyro" in currFirstWord.lower():
                    currSensor = "Gyroscope"
                else:
                    currSensor = "Accelerometer"
                if firstSensor == currSensor:
                    counter += 1
                else:
                    break
            line = fileStream.readline()
            firstWord = line.split(', ')[0]
            if "accel" in firstWord.lower():
                secondSensor = "Accelerometer"

        ## USED FOR PLOTTING
        # firstSensor_unFormattedTime = data[0:counter, 1]
        # firstSensor_Time = np.arange(firstSensor_unFormattedTime.shape[0]) / 50.0

        # secondSensor_unFormattedTime = data[counter:, 1]
        # secondSensor_Time = np.arange(secondSensor_unFormattedTime.shape[0]) / 50.0

        # firstSensor_1 = data[:counter, 2]
        # firstSensor_2 = data[:counter, 3]
        # firstSensor_3 = data[:counter, 4]
        # secondSensor_1 = data[counter:, 2]
        # secondSensor_2 = data[counter:, 3]
        # secondSensor_3 = data[counter:, 4]

        # firstNtoIgnore = 20

        # fft_firstSensor1 = scipy.fftpack.fft(firstSensor_1)
        # fft_firstSensor1[0:firstNtoIgnore] = 0
        # fft_firstSensor2 = scipy.fftpack.fft(firstSensor_2)
        # fft_firstSensor2[0:firstNtoIgnore] = 0
        # fft_firstSensor3 = scipy.fftpack.fft(firstSensor_3)
        # fft_firstSensor3[0:firstNtoIgnore] = 0
        # fft_secondSensor1 = scipy.fftpack.fft(secondSensor_1)
        # fft_secondSensor1[0:firstNtoIgnore] = 0
        # fft_secondSensor2 = scipy.fftpack.fft(secondSensor_2)
        # fft_secondSensor2[0:firstNtoIgnore] = 0
        # fft_secondSensor3 = scipy.fftpack.fft(secondSensor_3)
        # fft_secondSensor3[0:firstNtoIgnore] = 0
        #
        # x_firstSensor1 = np.linspace(0, 25, fft_firstSensor1.size / 2)
        # x_firstSensor2 = np.linspace(0, 25, fft_firstSensor2.size / 2)
        # x_firstSensor3 = np.linspace(0, 25, fft_firstSensor3.size / 2)
        # x_secondSensor1 = np.linspace(0, 25, fft_secondSensor1.size / 2)
        # x_secondSensor2 = np.linspace(0, 25, fft_secondSensor2.size / 2)
        # x_secondSensor3 = np.linspace(0, 25, fft_secondSensor3.size / 2)

        
        # 6 is the number of sensors, finalMatrix corresponds to FFT features, secondaryMatrix corresponds to statistical
        finalMatrix = np.zeros((((50 * fftWidth) / 2), numColumns, 6, totalNumOfLines / (50 * 2 * numSecondsPerImage)))
        secondaryMatrix = np.zeros((16, numColumns, 6, totalNumOfLines / (50 * 2 * numSecondsPerImage)))
        
        for n in range(finalMatrix.shape[3]):
            for j in range(finalMatrix.shape[1]):
                for k in range(3):
                    currFirstSensor = data[(n * 50 * numSecondsPerImage) + (j * 50 * fftJump): \
                        (n * 50 * numSecondsPerImage) + ((j * 50 * fftJump) + (50 * fftWidth)), k + 2]
                    currFirstSensor = currFirstSensor.astype(float)
                    curr_fft_firstSensor = scipy.fftpack.fft(currFirstSensor)
                    currSecondSensor = data[counter + (n * 50 * numSecondsPerImage) + (j * 50 * fftJump): \
                        counter + (n * 50 * numSecondsPerImage) + ((j * 50 * fftJump) + (50 * fftWidth)), k + 2]
                    currSecondSensor = currSecondSensor.astype(float)
                    curr_fft_secondSensor = scipy.fftpack.fft(currSecondSensor)
                    if "gyro" in firstSensor.lower():
                        finalMatrix[:, j, k, n] = np.abs(curr_fft_secondSensor[:curr_fft_secondSensor.size / 2])
                        finalMatrix[:, j, 3 + k, n] = np.abs(curr_fft_firstSensor[:curr_fft_firstSensor.size / 2])
                        secondaryMatrix[0, j, k, n] = np.mean(currSecondSensor)
                        secondaryMatrix[0, j, 3 + k, n] = np.mean(currFirstSensor)
                        secondaryMatrix[1, j, k, n] = np.std(currSecondSensor)
                        secondaryMatrix[1, j, 3 + k, n] = np.std(currFirstSensor)
                        secondaryMatrix[2, j, k, n] = secondaryMatrix[1, j, k, n] / float(secondaryMatrix[0, j, k, n])
                        secondaryMatrix[2, j, 3 + k, n] = secondaryMatrix[1, j, 3 + k, n] / float(
                            secondaryMatrix[0, j, 3 + k, n])
                        secondaryMatrix[3, j, k, n] = np.max(currSecondSensor) - np.min(currSecondSensor)
                        secondaryMatrix[3, j, 3 + k, n] = np.max(currFirstSensor) - np.min(currFirstSensor)
                        secondaryMatrix[4, j, k, n] = np.percentile(currSecondSensor, 10)
                        secondaryMatrix[4, j, 3 + k, n] = np.percentile(currFirstSensor, 10)
                        secondaryMatrix[5, j, k, n] = np.percentile(currSecondSensor, 25)
                        secondaryMatrix[5, j, 3 + k, n] = np.percentile(currFirstSensor, 25)
                        secondaryMatrix[6, j, k, n] = np.percentile(currSecondSensor, 50)
                        secondaryMatrix[6, j, 3 + k, n] = np.percentile(currFirstSensor, 50)
                        secondaryMatrix[7, j, k, n] = np.percentile(currSecondSensor, 75)
                        secondaryMatrix[7, j, 3 + k, n] = np.percentile(currFirstSensor, 75)
                        secondaryMatrix[8, j, k, n] = np.percentile(currSecondSensor, 90)
                        secondaryMatrix[8, j, 3 + k, n] = np.percentile(currFirstSensor, 90)
                        secondaryMatrix[9, j, k, n] = np.percentile(currSecondSensor, 75) - np.percentile(
                            currSecondSensor, 25)
                        secondaryMatrix[9, j, 3 + k, n] = np.percentile(currFirstSensor, 75) - np.percentile(
                            currFirstSensor, 25)
                        secondaryMatrix[10, j, k, n] = fehp.lag_one_autocorrelation(currSecondSensor)
                        secondaryMatrix[10, j, 3 + k, n] = fehp.lag_one_autocorrelation(currFirstSensor)
                        secondaryMatrix[11, j, k, n] = fehp.skewness(currSecondSensor)
                        secondaryMatrix[11, j, 3 + k, n] = fehp.skewness(currFirstSensor)
                        secondaryMatrix[12, j, k, n] = fehp.kurtosis(currSecondSensor)
                        secondaryMatrix[12, j, 3 + k, n] = fehp.kurtosis(currFirstSensor)
                        secondaryMatrix[13, j, k, n] = (np.linalg.norm(currSecondSensor)) ** 2
                        secondaryMatrix[13, j, 3 + k, n] = (np.linalg.norm(currFirstSensor)) ** 2
                        secondaryMatrix[14, j, k, n] = fehp.log_energy(currSecondSensor)
                        secondaryMatrix[14, j, 3 + k, n] = fehp.log_energy(currFirstSensor)
                        secondaryMatrix[15, j, k, n] = fehp.num_zero_crossings(currSecondSensor)
                        secondaryMatrix[15, j, 3 + k, n] = fehp.num_zero_crossings(currFirstSensor)
                    else:
                        finalMatrix[:, j, k, n] = np.abs(curr_fft_firstSensor[:curr_fft_firstSensor.size / 2])
                        finalMatrix[:, j, 3 + k, n] = np.abs(curr_fft_secondSensor[:curr_fft_secondSensor.size / 2])
                        secondaryMatrix[0, j, k, n] = np.mean(currFirstSensor)
                        secondaryMatrix[0, j, 3 + k, n] = np.mean(currSecondSensor)
                        secondaryMatrix[1, j, k, n] = np.std(currFirstSensor)
                        secondaryMatrix[1, j, 3 + k, n] = np.std(currSecondSensor)
                        secondaryMatrix[2, j, k, n] = secondaryMatrix[1, j, k, n] / float(secondaryMatrix[0, j, k, n])
                        secondaryMatrix[2, j, 3 + k, n] = secondaryMatrix[1, j, 3 + k, n] / float(
                            secondaryMatrix[0, j, 3 + k, n])
                        secondaryMatrix[3, j, k, n] = np.max(currFirstSensor) - np.min(currFirstSensor)
                        secondaryMatrix[3, j, 3 + k, n] = np.max(currSecondSensor) - np.min(currSecondSensor)
                        secondaryMatrix[4, j, k, n] = np.percentile(currFirstSensor, 10)
                        secondaryMatrix[4, j, 3 + k, n] = np.percentile(currSecondSensor, 10)
                        secondaryMatrix[5, j, k, n] = np.percentile(currFirstSensor, 25)
                        secondaryMatrix[5, j, 3 + k, n] = np.percentile(currSecondSensor, 25)
                        secondaryMatrix[6, j, k, n] = np.percentile(currFirstSensor, 50)
                        secondaryMatrix[6, j, 3 + k, n] = np.percentile(currSecondSensor, 50)
                        secondaryMatrix[7, j, k, n] = np.percentile(currFirstSensor, 75)
                        secondaryMatrix[7, j, 3 + k, n] = np.percentile(currSecondSensor, 75)
                        secondaryMatrix[8, j, k, n] = np.percentile(currFirstSensor, 90)
                        secondaryMatrix[8, j, 3 + k, n] = np.percentile(currSecondSensor, 90)
                        secondaryMatrix[9, j, k, n] = np.percentile(currFirstSensor, 75) - np.percentile(
                            currFirstSensor, 25)
                        secondaryMatrix[9, j, 3 + k, n] = np.percentile(currSecondSensor, 75) - np.percentile(
                            currSecondSensor, 25)
                        secondaryMatrix[10, j, k, n] = fehp.lag_one_autocorrelation(currFirstSensor)
                        secondaryMatrix[10, j, 3 + k, n] = fehp.lag_one_autocorrelation(currSecondSensor)
                        secondaryMatrix[11, j, k, n] = fehp.skewness(currFirstSensor)
                        secondaryMatrix[11, j, 3 + k, n] = fehp.skewness(currSecondSensor)
                        secondaryMatrix[12, j, k, n] = fehp.kurtosis(currFirstSensor)
                        secondaryMatrix[12, j, 3 + k, n] = fehp.kurtosis(currSecondSensor)
                        secondaryMatrix[13, j, k, n] = (np.linalg.norm(currFirstSensor)) ** 2
                        secondaryMatrix[13, j, 3 + k, n] = (np.linalg.norm(currSecondSensor)) ** 2
                        secondaryMatrix[14, j, k, n] = fehp.log_energy(currFirstSensor)
                        secondaryMatrix[14, j, 3 + k, n] = fehp.log_energy(currSecondSensor)
                        secondaryMatrix[15, j, k, n] = fehp.num_zero_crossings(currFirstSensor)
                        secondaryMatrix[15, j, 3 + k, n] = fehp.num_zero_crossings(currSecondSensor)

        print 'finalMatrix.shape: ', finalMatrix.shape

        outputIntermediateArray = np.reshape(finalMatrix, (finalMatrix.shape[3], -1))
        print 'outputIntermediateArray.shape: ', outputIntermediateArray.shape

        secondaryIntermediateArray = np.reshape(secondaryMatrix, (finalMatrix.shape[3], -1))

        #seventeenthFeatureMatrix is created to add the channel correlations to the statistical feature data
        seventeenthFeatureMatrix = np.zeros((15, numColumns, totalNumOfLines / (50 * 2 * numSecondsPerImage)))
        #     temp=np.zeros((50*fftWidth,numColumns,totalNumOfLines/(50*2*numSecondsPerImage)))

        for n in range(finalMatrix.shape[3]):
            for j in range(finalMatrix.shape[1]):
                inde = 0
                for k in range(3):
                    for l in range(k + 1, 3):
                        currFirstSensor1 = (data[(n * 50 * numSecondsPerImage) + (j * 50 * fftJump): \
                            (n * 50 * numSecondsPerImage) + ((j * 50 * fftJump) + (50 * fftWidth)), k + 2]).astype(
                            float)
                        currFirstSensor2 = (data[(n * 50 * numSecondsPerImage) + (j * 50 * fftJump): \
                            (n * 50 * numSecondsPerImage) + ((j * 50 * fftJump) + (50 * fftWidth)), l + 2]).astype(
                            float)
                        currSecondSensor1 = (data[counter + (n * 50 * numSecondsPerImage) + (j * 50 * fftJump): \
                            counter + (n * 50 * numSecondsPerImage) + ((j * 50 * fftJump) + (50 * fftWidth)),
                                             k + 2]).astype(float)
                        currSecondSensor2 = (data[counter + (n * 50 * numSecondsPerImage) + (j * 50 * fftJump): \
                            counter + (n * 50 * numSecondsPerImage) + ((j * 50 * fftJump) + (50 * fftWidth)),
                                             l + 2]).astype(float)

                        seventeenthFeatureMatrix[inde, j, n] = fehp.correlation(currFirstSensor1, currFirstSensor2)
                        inde += 1
                        seventeenthFeatureMatrix[inde, j, n] = fehp.correlation(currSecondSensor1, currSecondSensor2)
                        inde += 1
                        seventeenthFeatureMatrix[inde, j, n] = fehp.correlation(currSecondSensor1, currFirstSensor2)
                        inde += 1
                        seventeenthFeatureMatrix[inde, j, n] = fehp.correlation(currSecondSensor2, currFirstSensor1)
                        inde += 1
                        if k == 0 and l == 2:
                            seventeenthFeatureMatrix[inde, j, n] = fehp.correlation(currSecondSensor2, currFirstSensor2)
                            inde += 1
                        else:
                            seventeenthFeatureMatrix[inde, j, n] = fehp.correlation(currSecondSensor1, currFirstSensor1)
                            inde += 1

        print 'seventeenthFeatureMatrix.shape: ', seventeenthFeatureMatrix.shape
        outputThirdIntermediateArray = np.reshape(seventeenthFeatureMatrix, (finalMatrix.shape[3], -1))
        secondaryIntermediateArray = np.concatenate((secondaryIntermediateArray, outputThirdIntermediateArray), axis=1)

        intermediateOutputLabels = np.zeros((finalMatrix.shape[3], len(sports)))
        intermediateOutputLabels[:, ind] = 1
        print 'intermediateOutputLabels.shape for', sports[ind], ': ', intermediateOutputLabels.shape

        if ind == 0:
            outputLabels = intermediateOutputLabels
            outputArray = outputIntermediateArray
            outputSecondaryArray = secondaryIntermediateArray
        else:
            outputLabels = np.concatenate((outputLabels, intermediateOutputLabels))
            outputArray = np.concatenate((outputArray, outputIntermediateArray))
            outputSecondaryArray = np.concatenate((outputSecondaryArray, secondaryIntermediateArray))

    print '\noutputArray.shape: ', outputArray.shape
    print 'outputSecondaryArray.shape: ', outputSecondaryArray.shape
    print 'outputLabels.shape: ', outputLabels.shape

    np.savetxt('../Data/featuresFinal.csv', outputArray, delimiter=', ')
    np.savetxt('../Data/secondaryFeaturesFinal.csv', outputSecondaryArray, delimiter=', ')
    np.savetxt('../Data/labelsFinal.csv', outputLabels, fmt='%d', delimiter=', ')

################################################################
## PLOTTING

# import matplotlib.pyplot as plt
# %matplotlib inline
#
# for ind in range(len(sports)):
#     plt.figure(ind+1)
#     plt.plot(firstSensor_Time, firstSensor_1, 'r-',
#              firstSensor_Time, firstSensor_2, 'g-',
#              firstSensor_Time, firstSensor_3, 'b-', alpha=0.5)
#     plt.title(firstSensor + ' Signal')
#     plt.gcf().set_size_inches(15, 5)
#     plt.xlabel('Time (s)')
#     plt.ylabel('Value (m/s^2)')
#     plt.figure(ind+2)
#     plt.plot(secondSensor_Time, secondSensor_1, 'r-',
#              secondSensor_Time, secondSensor_2, 'g-',
#              secondSensor_Time, secondSensor_3, 'b-', alpha=0.5)
#     plt.gcf().set_size_inches(15, 5)
#     plt.title(secondSensor + ' Signal')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Value (m/s^2)')
#
#     plt.figure(ind+3)
#     plt.plot(x_firstSensor1, np.abs(fft_firstSensor1[:fft_firstSensor1.size / 2]), 'r-',
#              x_firstSensor2, np.abs(fft_firstSensor2[:fft_firstSensor2.size / 2]), 'g-',
#              x_firstSensor3, np.abs(fft_firstSensor3[:fft_firstSensor3.size / 2]), 'b-', alpha=0.5)
#     plt.gcf().set_size_inches(15, 5)
#     plt.title('FFT for ' + firstSensor)
#     plt.xlabel('Frequency')
#     plt.ylabel('Power')
#     plt.figure(ind+4)
#     plt.plot(x_secondSensor1, np.abs(fft_secondSensor1[:fft_secondSensor1.size / 2]), 'r-',
#              x_secondSensor2, np.abs(fft_secondSensor2[:fft_secondSensor2.size / 2]), 'g-',
#              x_secondSensor3, np.abs(fft_secondSensor3[:fft_secondSensor3.size / 2]), 'b-', alpha=0.5)
#     plt.gcf().set_size_inches(15, 5)
#     plt.title('FFT for ' + secondSensor)
#     plt.xlabel('Frequency')
#     plt.ylabel('Power')
#
#     # plt.figure(3)
#     # plt.plot(x_firstSensor1, np.abs(fft_firstSensor1[:fft_firstSensor1.size/2]), 'r-')
#     # plt.gcf().set_size_inches(15,5)
#     # plt.title('FFT of '+firstSensor+' 1')
#     # plt.xlabel('Frequency')
#     # plt.ylabel('Value')
#     # plt.figure(4)
#     # plt.plot(x_firstSensor2, np.abs(fft_firstSensor2[:fft_firstSensor2.size/2]), 'r-')
#     # plt.gcf().set_size_inches(15,5)
#     # plt.title('FFT of '+firstSensor+' 2')
#     # plt.xlabel('Frequency')
#     # plt.ylabel('Value')
#     # plt.figure(5)
#     # plt.plot(x_firstSensor3, np.abs(fft_firstSensor3[:fft_firstSensor3.size/2]), 'r-')
#     # plt.gcf().set_size_inches(15,5)
#     # plt.title('FFT of '+firstSensor+' 3')
#     # plt.xlabel('Frequency')
#     # plt.ylabel('Value')
#     # plt.figure(6)
#     # plt.plot(x_secondSensor1, np.abs(fft_secondSensor1[:fft_secondSensor1.size/2]), 'r-')
#     # plt.gcf().set_size_inches(15,5)
#     # plt.title('FFT of '+secondSensor+' 1')
#     # plt.xlabel('Frequency')
#     # plt.ylabel('Value')
#     # plt.figure(7)
#     # plt.plot(x_secondSensor2, np.abs(fft_secondSensor2[:fft_secondSensor2.size/2]), 'r-')
#     # plt.gcf().set_size_inches(15,5)
#     # plt.title('FFT of '+secondSensor+' 2')
#     # plt.xlabel('Frequency')
#     # plt.ylabel('Value')
#     # plt.figure(8)
#     # plt.plot(x_secondSensor3, np.abs(fft_secondSensor3[:fft_secondSensor3.size/2]), 'r-')
#     # plt.gcf().set_size_inches(15,5)
#     # plt.title('FFT of '+secondSensor+' 3')
#     # plt.xlabel('Frequency')
#     # plt.ylabel('Value')
#     #
#     # for i in range(finalMatrix.shape[3]):
#     #     plt.rcParams['figure.figsize'] = (15, 4)
#     #     plt.figure(9+i)
#     #     plt.subplot(161)
#     #     plt.title('A1')
#     #     plt.xlabel('Time')
#     #     plt.ylabel('Frequency')
#     #     plt.yticks( np.arange(5*fftWidth,26*fftWidth,5*fftWidth), (5,10,15,20,25) )
#     #     plt.imshow(finalMatrix[:,:,0,i], aspect='auto')
#     #     #plt.colorbar()
#     #     plt.subplot(162)
#     #     plt.title('A2')
#     #     plt.xlabel('Time')
#     #     plt.yticks( np.arange(5*fftWidth,26*fftWidth,5*fftWidth), (5,10,15,20,25) )
#     #     plt.imshow(finalMatrix[:,:,1,i], aspect='auto')
#     #     #plt.colorbar()
#     #     plt.subplot(163)
#     #     plt.title('A3')
#     #     plt.xlabel('Time')
#     #     plt.yticks( np.arange(5*fftWidth,26*fftWidth,5*fftWidth), (5,10,15,20,25) )
#     #     plt.imshow(finalMatrix[:,:,2,i], aspect='auto')
#     #     #plt.colorbar()
#     #     plt.subplot(164)
#     #     plt.title('G1')
#     #     plt.xlabel('Time')
#     #     plt.yticks( np.arange(5*fftWidth,26*fftWidth,5*fftWidth), (5,10,15,20,25) )
#     #     plt.imshow(finalMatrix[:,:,3,i], aspect='auto')
#     #     #plt.colorbar()
#     #     plt.subplot(165)
#     #     plt.title('G2')
#     #     plt.xlabel('Time')
#     #     plt.yticks( np.arange(5*fftWidth,26*fftWidth,5*fftWidth), (5,10,15,20,25) )
#     #     plt.imshow(finalMatrix[:,:,4,i], aspect='auto')
#     #     #plt.colorbar()
#     #     plt.subplot(166)
#     #     plt.title('G3')
#     #     plt.xlabel('Time')
#     #     plt.yticks( np.arange(5*fftWidth,26*fftWidth,5*fftWidth), (5,10,15,20,25) )
#     #     plt.imshow(finalMatrix[:,:,5,i], aspect='auto')
#     #     #plt.colorbar()
#
# plt.show()

################################################################