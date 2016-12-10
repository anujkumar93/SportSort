import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import glob
import os

def data_preprocessing(sports=['Badminton','Basketball','Foosball','Running','Skating','Walking'],
                       secondsToKeep=30,
                       trimLength=15,
                       newPerson=False):
    """
    Pre-processes the raw data files to make sure the data from both sensors is of the same size.
    Discards some amount of raw data at the start and end of the files to remove miscellaneous activity
    that gets recorded while starting and ending data collection.
    Then divides the data into integral samples of given size.
    :param sports: List of sport names involved
    :param secondsToKeep: Number of seconds to keep in each sample
    :param trimLength: Number of seconds to trim at the start and enf of a raw data file
    :param newPerson: Set True to generate unseen test data
    :return: Nothing
    """
    for sport in sports:
        source_dir      = '../Data/' + sport
        finalOutputFile = '../Data/' + sport + '/Final.csv'
        if newPerson:
            finalOutputFile = '../Data/' + sport + '/newPersonFinal.csv'
        outputFileAcc   = finalOutputFile[:len(finalOutputFile)-4]+'_Acc'+finalOutputFile[len(finalOutputFile)-4:]
        outputFileGyro  = finalOutputFile[:len(finalOutputFile)-4]+'_Gyro'+finalOutputFile[len(finalOutputFile)-4:]

        file_list = glob.glob(source_dir + '/*.csv')

        with open(outputFileAcc,'w') as outFileAcc:
            with open(outputFileGyro,'w') as outFileGyro:
                for i in range(len(file_list)):
                    # avoiding files with 'Final' in its name
                    if 'Final' in file_list[i]:
                        continue

                    if newPerson:
                        # avoiding files with 'newPerson' NOT in its name
                        if 'newPerson' not in file_list[i]:
                            continue
                    else:
                        # avoiding files with 'newPerson' in its name
                        if 'newPerson' in file_list[i]:
                            continue

                    with open(file_list[i], 'r') as fileStream:
                        counter=1
                        firstSensor  = "Accelerometer"
                        secondSensor = "Gyroscope"

                        print '\nFile:', file_list[i]

                        firstLine=fileStream.readline()
                        firstWord=firstLine.split(', ')[0]

                        if "gyro" in firstWord.lower():
                            firstSensor = "Gyroscope"

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
                            secondSensor = "Accelerometer"

                        fileStream.seek(0)
                        lines           = fileStream.readlines()
                        totalNumOfLines = len(lines)
                        secondCounter   = totalNumOfLines-counter

                        print '1st sensor samples          =', counter
                        print '2nd sensor samples          =', secondCounter

                        # break data into two sets - for two sensors
                        firstSensorLines  = lines[:counter] # size = counter
                        secondSensorLines = lines[counter:] # size = secondCounter

                        if secondCounter < counter:
                            # subsampling of larger counted sensor data
                            diff = counter - secondCounter
                            step = counter / diff
                            toDelete = np.zeros(diff)
                            for j in range(diff):
                                toDelete[j] = int(np.floor(j*step))

                            subsample_lineNum = np.delete(np.arange(counter,dtype=np.int32), toDelete)
                            subsample_lines=[lines[ind] for ind in subsample_lineNum]
                            # trimming
                            if len(subsample_lines) > 2*50*trimLength:
                                subsample_lines = subsample_lines[50*trimLength:len(subsample_lines)-50*trimLength]
                                secondSensorLines = secondSensorLines[50*trimLength:len(secondSensorLines)-50*trimLength]

                            # saving
                            linesToKeep = len(subsample_lines)-(len(subsample_lines)%(50*secondsToKeep))
                            finalCounterLines=subsample_lines[0:linesToKeep]
                            finalSecondCounterLines=secondSensorLines[0:linesToKeep]

                        elif secondCounter > counter:
                            diff = secondCounter - counter
                            step = secondCounter / diff
                            toDelete = np.zeros(diff)
                            for j in range(diff):
                                toDelete[j] = int(np.floor(j*step))

                            subsample_lineNum = np.delete(np.arange(secondCounter,dtype=np.int32), toDelete)
                            subsample_lines=[lines[counter+ind] for ind in subsample_lineNum]

                            if len(subsample_lines) > 2*50*trimLength:
                                firstSensorLines = firstSensorLines[50*trimLength:len(firstSensorLines)-50*trimLength]
                                subsample_lines = subsample_lines[50*trimLength:len(subsample_lines)-50*trimLength]

                            linesToKeep = len(subsample_lines)-(len(subsample_lines)%(50*secondsToKeep))
                            finalCounterLines=firstSensorLines[0:linesToKeep]
                            finalSecondCounterLines=subsample_lines[0:linesToKeep]

                        else:
                            if len(firstSensorLines) > 2*50*trimLength:
                                firstSensorLines = firstSensorLines[50*trimLength:len(firstSensorLines)-50*trimLength]
                                secondSensorLines = secondSensorLines[50*trimLength:len(secondSensorLines)-50*trimLength]

                            linesToKeep = len(secondSensorLines)-(len(secondSensorLines)%(50*secondsToKeep))
                            finalCounterLines=firstSensorLines[:linesToKeep]
                            finalSecondCounterLines=secondSensorLines[:linesToKeep]

                        print 'Adjusted 1st sensor samples =', len(finalCounterLines)
                        print 'Adjusted 2nd sensor samples =', len(finalSecondCounterLines)

                        if firstSensor == "Accelerometer":
                            for item in finalCounterLines:
                                outFileAcc.write("%s" % item)
                            for item in finalSecondCounterLines:
                                outFileGyro.write("%s" % item)
                        elif firstSensor == "Gyroscope":
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

        os.remove(outputFileAcc)
        os.remove(outputFileGyro)
