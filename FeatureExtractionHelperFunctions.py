import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import os


def file_len(fname):
    """
    Counts and returns the number of lines in a file
    :param fname: file path
    :return: number of lines in the file (int)
    """
    if os.path.exists(fname) and os.path.getsize(fname) > 0:
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1
    else:
        return 0

def lag_one_autocorrelation(arra):
    denom=((np.std(arra))**2)*arra.size
    summ=0
    mean=np.mean(arra)
    for i in range(arra.size-1):
        summ+=(arra[i]-mean)*(arra[i+1]-mean)
    return summ/float(denom)

def skewness(arra):
    denom=(np.std(arra))**3
    num=0
    mean=np.mean(arra)
    for i in range(arra.size):
        num+=(arra[i]-mean)**3
    num=num/float(arra.size)
    return num/denom

def kurtosis(arra):
    denom=(np.std(arra))**6
    num=0
    mean=np.mean(arra)
    for i in range(arra.size):
        num+=(arra[i]-mean)**4
    num=num/float(arra.size)
    return (num/denom)-3

def log_energy(arra):
    summ=0
    for i in range(arra.size):
        if (arra[i]==0):
            pass
#             summ+=np.log(0.01)
        else:
            summ+=np.log(arra[i]**2)
    return summ

def num_zero_crossings(arra):
    arra2=np.copy(arra)
    arra2=arra2-np.mean(arra2)
    return (np.where(np.diff(np.sign(arra2)))[0]).size

def correlation(arra, arra2):
    mean1=np.mean(arra)
    mean2=np.mean(arra2)
    num=0
    denom1=((np.std(arra))**2)*arra.size
    denom2=((np.std(arra2))**2)*arra2.size
    for i in range(arra.size):
        num+=(arra[i]-mean1)*(arra2[i]-mean2)
    return float(num)/(denom1*denom2)