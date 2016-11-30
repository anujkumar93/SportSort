import numpy as np
import scipy.ndimage as sn
import scipy.fftpack
import datetime
import time
import random
import glob
import matplotlib.pyplot as plt
%matplotlib inline

sports = ['Walking','Running','Skating','Badminton','Basketball']

for sport in sports:
    source_dir='../Data/' + sport
    finalOutputFile='../Data/' + sport + 'Final.csv'
    outputFileAcc=finalOutputFile[:len(finalOutputFile)-4]+'_Acc'+finalOutputFile[len(finalOutputFile)-4:]
    outputFileGyro=finalOutputFile[:len(finalOutputFile)-4]+'_Gyro'+finalOutputFile[len(finalOutputFile)-4:]

    secondsToKeep=30

    file_list = glob.glob(source_dir + '/*.csv')

    with open(outputFileAcc,'w') as outFileAcc:
        with open(outputFileGyro,'w') as outFileGyro:
            for i in range(len(file_list)):
                with open(file_list[i], 'r') as fileStream:
                    counter=1
                    firstSensor="Accelerometer"
                    secondSensor="Gyroscope"
                    
                    print '\nFile:', file_list[i]
                    
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
                    fileStream.seek(0)
                    lines=fileStream.readlines()
                    totalNumOfLines=len(lines)
                    secondCounter=totalNumOfLines-counter
                    
                    print '1st sensor samples          =', counter
                    print '2nd sensor samples          =', secondCounter
                    
                    if secondCounter<counter:
                        #random_lineNum=random.sample(range(counter), secondCounter)
                        #random_lineNum.sort()
                        #random_lines=[lines[i] for i in random_lineNum]
                        diff = counfloorter - secondCounter
                        step = counter / diff
                        toDelete = np.zeros(diff)
                        for i in range(diff):
                            toDelete[i] = int(np.floor(i*step))
                        subsample_lineNum = np.delete(np.arange(counter,dtype=np.int32), toDelete)
                        subsample_lines=[lines[i] for i in subsample_lineNum]
                        finalCounterLines=subsample_lines[0:secondCounter-(secondCounter%(50*secondsToKeep))]
                        finalSecondCounterLines=lines[counter:counter+(secondCounter-(secondCounter%(50*secondsToKeep)))]
                    elif secondCounter>counter:
                        #random_lineNum=random.sample(range(secondCounter), counter)
                        #random_lineNum.sort()
                        #random_lines=[lines[counter+i] for i in random_lineNum]
                        diff = secondCounter - counter
                        step = np.floor(secondCounter / diff)
                        toDelete = np.zeros(diff)
                        for i in range(diff):
                            toDelete[i] = int(np.floor(i*step))
                        subsample_lineNum = np.delete(np.arange(secondCounter,dtype=np.int32), toDelete)
                        subsample_lines=[lines[counter+i] for i in subsample_lineNum]
                        finalCounterLines=lines[:counter-(counter%(50*secondsToKeep))]
                        finalSecondCounterLines=subsample_lines[0:counter-(counter%(50*secondsToKeep))]
                    else:
                        finalCounterLines=lines[:counter-(counter%(50*secondsToKeep))]
                        finalSecondCounterLines=lines[counter:counter+(secondCounter-(secondCounter%(50*secondsToKeep)))]
                    trimLength = 15
                    if len(finalCounterLines) > 2*50*trimLength and len(finalSecondCounterLines) > 2*50*trimLength:
                        finalCounterLines = finalCounterLines[50*trimLength:len(finalCounterLines)-50*trimLength]
                        finalSecondCounterLines = finalSecondCounterLines[50*trimLength:len(finalSecondCounterLines)-50*trimLength]
                    else:
                        finalCounterLines = []
                        finalSecondCounterLines = []
                        
                    print 'Adjusted 1st sensor samples =', len(finalCounterLines)
                    print 'Adjusted 2nd sensor samples =', len(finalSecondCounterLines)
                    
                    if (firstSensor=="Accelerometer"):
                        for item in finalCounterLines:
                            outFileAcc.write("%s" % item)
                        for item in finalSecondCounterLines:
                            outFileGyro.write("%s" % item)
                    elif (firstSensor=="Gyroscope"):
                        for item in finalSecondCounterLines:
                            outFileAcc.write("%s" % item)
                        for item in finalCounterLines:
                            outFileGyro.write("%s" % item)

    with open(outputFileAcc,'r') as inFileAcc:
        with open(outputFileGyro,'r') as inFileGyro:
            with open(finalOutputFile, 'w') as outFile:
                lines=inFileAcc.readlines()
                for item in lines:
                    outFile.write("%s" % item)
                lines=inFileGyro.readlines()
                for item in lines:
                    outFile.write("%s" % item)
      
