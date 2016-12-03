import numpy as np
import scipy.ndimage as sn
import scipy.fftpack
import datetime
import time
import matplotlib.pyplot as plt
%matplotlib inline

def file_len(fname):
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1

sports = ['Badminton','Basketball','Running','Skating','Walking']

fftWidth = 6             # no of seconds to take FFT over
fftJump  = 2             # no of seconds to jump to take next fft
numSecondsPerImage = 30  #make sure this divides 60 exactly  (change in script)
numColumns=(numSecondsPerImage-fftWidth)/fftJump+1    #total no of columns to include in one image

for ind in range(len(sports)):
    fileName='./Data/' + sports[ind] + 'Final.csv'

# fileName='./Data/WalkingFinal.csv'
# fileName='./Data/RunningFinal.csv'
# fileName='./Data/SkatingFinal.csv'
# fileName='./Data/BadmintonFinal.csv'
# fileName='./Data/BasketballFinal.csv'

    data=np.loadtxt(fileName,dtype='string', delimiter=', ')
    counter=1;
    firstSensor="Accelerometer"
    secondSensor="Gyroscope"
    
    totalNumOfLines=file_len(fileName)

    with open(fileName, 'r') as fileStream:
        firstLine=fileStream.readline()
        firstWord=firstLine.split(', ')[0]
        if "gyro" in firstWord.lower():
            firstSensor="Gyroscope"
        while True:
            line=fileStream.readline()
            currFirstWord=line.split(', ')[0]
            if "gyro" in currFirstWord.lower():
                currSensor="Gyroscope"
            else:
                currSensor="Accelerometer"
            if firstSensor==currSensor:
                counter+=1
            else:
                break
        line=fileStream.readline()
        firstWord=line.split(', ')[0]
        if "accel" in firstWord.lower():
            secondSensor="Accelerometer"

    firstSensor_unFormattedTime=data[0:counter,1]
    firstSensor_Time=np.arange(firstSensor_unFormattedTime.shape[0]) / 50.0
    
    secondSensor_unFormattedTime=data[counter:,1]
    secondSensor_Time=np.arange(secondSensor_unFormattedTime.shape[0]) / 50.0

    firstSensor_1=data[:counter,2]
    firstSensor_2=data[:counter,3]
    firstSensor_3=data[:counter,4]
    secondSensor_1=data[counter:,2]
    secondSensor_2=data[counter:,3]
    secondSensor_3=data[counter:,4]

    firstNtoIgnore = 20

    fft_firstSensor1=scipy.fftpack.fft(firstSensor_1)
    fft_firstSensor1[0:firstNtoIgnore]=0
    fft_firstSensor2=scipy.fftpack.fft(firstSensor_2)
    fft_firstSensor2[0:firstNtoIgnore]=0
    fft_firstSensor3=scipy.fftpack.fft(firstSensor_3)
    fft_firstSensor3[0:firstNtoIgnore]=0
    fft_secondSensor1=scipy.fftpack.fft(secondSensor_1)
    fft_secondSensor1[0:firstNtoIgnore]=0
    fft_secondSensor2=scipy.fftpack.fft(secondSensor_2)
    fft_secondSensor2[0:firstNtoIgnore]=0
    fft_secondSensor3=scipy.fftpack.fft(secondSensor_3)
    fft_secondSensor3[0:firstNtoIgnore]=0

    x_firstSensor1=np.linspace(0,25,fft_firstSensor1.size/2)
    x_firstSensor2=np.linspace(0,25,fft_firstSensor2.size/2)
    x_firstSensor3=np.linspace(0,25,fft_firstSensor3.size/2)
    x_secondSensor1=np.linspace(0,25,fft_secondSensor1.size/2)
    x_secondSensor2=np.linspace(0,25,fft_secondSensor2.size/2)
    x_secondSensor3=np.linspace(0,25,fft_secondSensor3.size/2)

    #6 is the number of sensors
    finalMatrix=np.zeros((((50*fftWidth)/2),numColumns,6,totalNumOfLines/(50*2*numSecondsPerImage)))
    for n in range(finalMatrix.shape[3]):
        for k in range(3):
            for j in range(finalMatrix.shape[1]):
                currFirstSensor=data[(n*50*numSecondsPerImage)+(j*50*fftJump):\
                                     (n*50*numSecondsPerImage)+((j*50*fftJump)+(50*fftWidth)),k+2]
                curr_fft_firstSensor=scipy.fftpack.fft(currFirstSensor)
                currSecondSensor=data[counter+(n*50*numSecondsPerImage)+(j*50*fftJump):\
                                      counter+(n*50*numSecondsPerImage)+((j*50*fftJump)+(50*fftWidth)),k+2]
                curr_fft_secondSensor=scipy.fftpack.fft(currSecondSensor)
                if "gyro" in firstSensor.lower():
                    finalMatrix[:,j,k,n]=np.abs(curr_fft_secondSensor[:curr_fft_secondSensor.size/2])
                    finalMatrix[:,j,3+k,n]=np.abs(curr_fft_firstSensor[:curr_fft_firstSensor.size/2])
                else:
                    finalMatrix[:,j,k,n]=np.abs(curr_fft_firstSensor[:curr_fft_firstSensor.size/2])
                    finalMatrix[:,j,3+k,n]=np.abs(curr_fft_secondSensor[:curr_fft_secondSensor.size/2])

    print 'finalMatrix.shape for', sports[ind], ': ', finalMatrix.shape
    
    outputIntermediateArray=np.reshape(finalMatrix,(finalMatrix.shape[3],-1))
    print 'outputArray.shape for', sports[ind], ': ', outputIntermediateArray.shape
    
    intermediateOutputLabels=np.zeros((finalMatrix.shape[3],len(sports)))
    intermediateOutputLabels[:,ind]=1
    print 'outputLabels.shape for', sports[ind], ': ', intermediateOutputLabels.shape
    
    if ind==0:
        outputLabels=intermediateOutputLabels
        outputArray=outputIntermediateArray
    else:
        outputLabels=np.concatenate((outputLabels, intermediateOutputLabels))
        outputArray=np.concatenate((outputArray, outputIntermediateArray))
        
    print 'outputArray.shape: ', outputArray.shape
    print 'outputLabels.shape: ', outputLabels.shape

    plt.figure(1)
    plt.plot(firstSensor_Time, firstSensor_1, 'r-', 
             firstSensor_Time, firstSensor_2, 'g-', 
             firstSensor_Time, firstSensor_3, 'b-', alpha=0.5)
    plt.title(firstSensor+' Signal')
    plt.gcf().set_size_inches(15,5)
    plt.xlabel('Time (s)')
    plt.ylabel('Value (m/s^2)')
    plt.figure(2)
    plt.plot(secondSensor_Time, secondSensor_1, 'r-', 
             secondSensor_Time, secondSensor_2, 'g-', 
             secondSensor_Time, secondSensor_3, 'b-', alpha=0.5)
    plt.gcf().set_size_inches(15,5)
    plt.title(secondSensor+' Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Value (m/s^2)')

    plt.figure(3)
    plt.plot(x_firstSensor1, np.abs(fft_firstSensor1[:fft_firstSensor1.size/2]), 'r-', 
             x_firstSensor2, np.abs(fft_firstSensor2[:fft_firstSensor2.size/2]), 'g-', 
             x_firstSensor3, np.abs(fft_firstSensor3[:fft_firstSensor3.size/2]), 'b-', alpha=0.5)
    plt.gcf().set_size_inches(15,5)
    plt.title('FFT for '+firstSensor)
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.figure(4)
    plt.plot(x_secondSensor1, np.abs(fft_secondSensor1[:fft_secondSensor1.size/2]), 'r-', 
             x_secondSensor2, np.abs(fft_secondSensor2[:fft_secondSensor2.size/2]), 'g-', 
             x_secondSensor3, np.abs(fft_secondSensor3[:fft_secondSensor3.size/2]), 'b-', alpha=0.5)
    plt.gcf().set_size_inches(15,5)
    plt.title('FFT for '+secondSensor)
    plt.xlabel('Frequency')
    plt.ylabel('Power')

#     plt.figure(3)
#     plt.plot(x_firstSensor1, np.abs(fft_firstSensor1[:fft_firstSensor1.size/2]), 'r-')
#     plt.gcf().set_size_inches(15,5)
#     plt.title('FFT of '+firstSensor+' 1')
#     plt.xlabel('Frequency')
#     plt.ylabel('Value')
#     plt.figure(4)
#     plt.plot(x_firstSensor2, np.abs(fft_firstSensor2[:fft_firstSensor2.size/2]), 'r-')
#     plt.gcf().set_size_inches(15,5)
#     plt.title('FFT of '+firstSensor+' 2')
#     plt.xlabel('Frequency')
#     plt.ylabel('Value')
#     plt.figure(5)
#     plt.plot(x_firstSensor3, np.abs(fft_firstSensor3[:fft_firstSensor3.size/2]), 'r-')
#     plt.gcf().set_size_inches(15,5)
#     plt.title('FFT of '+firstSensor+' 3')
#     plt.xlabel('Frequency')
#     plt.ylabel('Value')
#     plt.figure(6)
#     plt.plot(x_secondSensor1, np.abs(fft_secondSensor1[:fft_secondSensor1.size/2]), 'r-')
#     plt.gcf().set_size_inches(15,5)
#     plt.title('FFT of '+secondSensor+' 1')
#     plt.xlabel('Frequency')
#     plt.ylabel('Value')
#     plt.figure(7)
#     plt.plot(x_secondSensor2, np.abs(fft_secondSensor2[:fft_secondSensor2.size/2]), 'r-')
#     plt.gcf().set_size_inches(15,5)
#     plt.title('FFT of '+secondSensor+' 2')
#     plt.xlabel('Frequency')
#     plt.ylabel('Value')
#     plt.figure(8)
#     plt.plot(x_secondSensor3, np.abs(fft_secondSensor3[:fft_secondSensor3.size/2]), 'r-')
#     plt.gcf().set_size_inches(15,5)
#     plt.title('FFT of '+secondSensor+' 3')
#     plt.xlabel('Frequency')
#     plt.ylabel('Value')

#     for i in range(finalMatrix.shape[3]):
#         plt.rcParams['figure.figsize'] = (15, 4)
#         plt.figure(9+i)
#         plt.subplot(161)
#         plt.title('A1')
#         plt.xlabel('Time')
#         plt.ylabel('Frequency')
#         plt.yticks( np.arange(5*fftWidth,26*fftWidth,5*fftWidth), (5,10,15,20,25) )
#         plt.imshow(finalMatrix[:,:,0,i], aspect='auto')
#         #plt.colorbar()
#         plt.subplot(162)
#         plt.title('A2')
#         plt.xlabel('Time')
#         plt.yticks( np.arange(5*fftWidth,26*fftWidth,5*fftWidth), (5,10,15,20,25) )
#         plt.imshow(finalMatrix[:,:,1,i], aspect='auto')
#         #plt.colorbar()
#         plt.subplot(163)
#         plt.title('A3')
#         plt.xlabel('Time')
#         plt.yticks( np.arange(5*fftWidth,26*fftWidth,5*fftWidth), (5,10,15,20,25) )
#         plt.imshow(finalMatrix[:,:,2,i], aspect='auto')
#         #plt.colorbar()
#         plt.subplot(164)
#         plt.title('G1')
#         plt.xlabel('Time')
#         plt.yticks( np.arange(5*fftWidth,26*fftWidth,5*fftWidth), (5,10,15,20,25) )
#         plt.imshow(finalMatrix[:,:,3,i], aspect='auto')
#         #plt.colorbar()
#         plt.subplot(165)
#         plt.title('G2')
#         plt.xlabel('Time')
#         plt.yticks( np.arange(5*fftWidth,26*fftWidth,5*fftWidth), (5,10,15,20,25) )
#         plt.imshow(finalMatrix[:,:,4,i], aspect='auto')
#         #plt.colorbar()
#         plt.subplot(166)
#         plt.title('G3')
#         plt.xlabel('Time')
#         plt.yticks( np.arange(5*fftWidth,26*fftWidth,5*fftWidth), (5,10,15,20,25) )
#         plt.imshow(finalMatrix[:,:,5,i], aspect='auto')
#         #plt.colorbar()

plt.show()
np.savetxt('./Data/featuresFinal.csv',outputArray, delimiter=', ')
np.savetxt('./Data/labelsFinal.csv',outputLabels, delimiter=', ')
