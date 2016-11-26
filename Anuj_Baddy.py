import numpy as np
import scipy.ndimage as sn
import scipy.fftpack
import datetime
import time
import matplotlib.pyplot as plt
%matplotlib inline

fileName='./ML project/Badminton_person2.csv'
data=np.loadtxt(fileName,dtype='string', delimiter=', ')
counter=1;
firstSensor="Accelerometer"
secondSensor="Gyroscope"

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

totalNumOfLines=file_len(fileName)

with open(fileName, 'r') as fileStream:
	firstLine=fileStream.readline()
	firstWord=firstLine.split(', ')[0]
	if "gyro" in firstWord.lower():
	    firstSensor="Gyroscope"
	while True:
	    line=fileStream.readline()
	    currFirstWord=line.split(', ')[0]
	    if firstWord==currFirstWord:
	        counter+=1
	    else:
	        break
	line=fileStream.readline()
	firstWord=line.split(', ')[0]
	if "accel" in firstWord.lower():
	    secondSensor="Accelerometer"

firstSensor_unFormattedTime=data[0:counter,1]
firstSensor_Time=np.zeros(firstSensor_unFormattedTime.shape[0])
for i in range(firstSensor_unFormattedTime.shape[0]):
    pt =datetime.datetime.strptime(firstSensor_unFormattedTime[i],'%b %d,%Y %H:%M:%S.%f')
    total_seconds = pt.second+pt.minute*60+pt.hour*3600
    firstSensor_Time[i]=total_seconds
secondSensor_unFormattedTime=data[counter:,1]
secondSensor_Time=np.zeros(secondSensor_unFormattedTime.shape[0])
for i in range(secondSensor_unFormattedTime.shape[0]):
    pt =datetime.datetime.strptime(secondSensor_unFormattedTime[i],'%b %d,%Y %H:%M:%S.%f')
    total_seconds = pt.second+pt.minute*60+pt.hour*3600
    secondSensor_Time[i]=total_seconds
    
firstSensor_1=data[:counter,2]
firstSensor_2=data[:counter,3]
firstSensor_3=data[:counter,4]
secondSensor_1=data[counter:,2]
secondSensor_2=data[counter:,3]
secondSensor_3=data[counter:,4]

fft_firstSensor1=scipy.fftpack.fft(firstSensor_1)
# fft_firstSensor1[0:1]=0
fft_firstSensor2=scipy.fftpack.fft(firstSensor_2)
# fft_firstSensor2[0:1]=0
fft_firstSensor3=scipy.fftpack.fft(firstSensor_3)
# fft_firstSensor3[0:1]=0
fft_secondSensor1=scipy.fftpack.fft(secondSensor_1)
# fft_secondSensor1[0:1]=0
fft_secondSensor2=scipy.fftpack.fft(secondSensor_2)
# fft_secondSensor2[0:1]=0
fft_secondSensor3=scipy.fftpack.fft(secondSensor_3)
# fft_secondSensor3[0:1]=0

x_firstSensor1=np.linspace(0,25,fft_firstSensor1.size/2)
x_firstSensor2=np.linspace(0,25,fft_firstSensor2.size/2)
x_firstSensor3=np.linspace(0,25,fft_firstSensor3.size/2)
x_secondSensor1=np.linspace(0,25,fft_secondSensor1.size/2)
x_secondSensor2=np.linspace(0,25,fft_secondSensor2.size/2)
x_secondSensor3=np.linspace(0,25,fft_secondSensor3.size/2)

fftWidth=5 #no of seconds to take FFT over
fftJump=1 #no of seconds to jump to take next fft
numColumns=26 #total no of columns to include in one image
numSecondsPerImage=fftJump*(numColumns-1)+fftWidth #make sure this divides 120 exactly (since concatenated file has 2 min)

#6 is the number of sensors
finalMatrix=np.zeros(((50*fftWidth)/2),numColumns,6,totalNumOfLines/(50*2*numSecondsPerImage))
for n in range(finalMatrix.shape[3]):
    for k in range(3):
        for j in range(finalMatrix.shape[1]):
            currFirstSensor=data[n*1750+250*j:counter,k+2]
            currSecondSensor=data[counter:,k+2]
            


plt.figure(1)
plt.plot(firstSensor_Time, firstSensor_1, 'r-',firstSensor_Time,firstSensor_2,'g-',firstSensor_Time,firstSensor_3,'b-')
plt.title(firstSensor+' Signal')
plt.gcf().set_size_inches(15,5)
plt.xlabel('Time')
plt.ylabel('Value')
plt.figure(2)
plt.plot(secondSensor_Time, secondSensor_1, 'r-',secondSensor_Time,secondSensor_2,'g-',secondSensor_Time,secondSensor_3,'b-')
plt.gcf().set_size_inches(15,5)
plt.title(secondSensor+' Signal')
plt.xlabel('Time')
plt.ylabel('Value')
plt.figure(3)
plt.plot(x_secondSensor1, np.abs(fft_secondSensor1[:fft_secondSensor1.size/2]), 'r-')
plt.gcf().set_size_inches(15,5)
plt.title('FFT of '+secondSensor+' 1')
plt.xlabel('Frequency')
plt.ylabel('Value')
plt.figure(4)
plt.plot(x_secondSensor2, np.abs(fft_secondSensor2[:fft_secondSensor2.size/2]), 'r-')
plt.gcf().set_size_inches(15,5)
plt.title('FFT of '+secondSensor+' 2')
plt.xlabel('Frequency')
plt.ylabel('Value')
plt.figure(5)
plt.plot(x_secondSensor3, np.abs(fft_secondSensor3[:fft_secondSensor3.size/2]), 'r-')
plt.gcf().set_size_inches(15,5)
plt.title('FFT of '+secondSensor+' 3')
plt.xlabel('Frequency')
plt.ylabel('Value')
plt.figure(6)
plt.plot(x_firstSensor1, np.abs(fft_firstSensor1[:fft_firstSensor1.size/2]), 'r-')
plt.gcf().set_size_inches(15,5)
plt.title('FFT of '+firstSensor+' 1')
plt.xlabel('Frequency')
plt.ylabel('Value')
plt.figure(7)
plt.plot(x_firstSensor2, np.abs(fft_firstSensor2[:fft_firstSensor2.size/2]), 'r-')
plt.gcf().set_size_inches(15,5)
plt.title('FFT of '+firstSensor+' 2')
plt.xlabel('Frequency')
plt.ylabel('Value')
plt.figure(8)
plt.plot(x_firstSensor3, np.abs(fft_firstSensor3[:fft_firstSensor3.size/2]), 'r-')
plt.gcf().set_size_inches(15,5)
plt.title('FFT of '+firstSensor+' 3')
plt.xlabel('Frequency')
plt.ylabel('Value')

plt.show()