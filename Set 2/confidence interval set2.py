import numpy as np
import math
import matplotlib as plt
import matplotlib.pyplot as plt
from mpmath import*
import os
'''
x is the array of parameters, and p is the array of probabilities as a function
of each of the elements in the parameter array. the size is the total width of
the confidence interval.
'''
def ConfidenceInterval(x, p, size):
    counts = np.empty(0)
    n = 10**3
#find peak
    peakIndex = np.argmax(p)
    peakProb= p[peakIndex]
    peakLocation = x[peakIndex]
    for i in range(len(p)):
        if(i==peakIndex):
#Compute the cdf by multiplying the probability of each parameter by 10^5
            countsPeakIndex = len(counts)+int(.5*int(p[i]*(n)))
        #record peak around which to set the confidence interval in the cdf
        count = [x[i]]*int(p[i]*(n))
#append the paramter that many times to the end of a singular 1D array
        counts = np.append(counts, count)
    arrayLength = len(counts)
    halfinterval = int(arrayLength*size/2)
#find lower bound
    leftParameter = counts[countsPeakIndex-halfinterval]
#find upperbound
    rightParameter = counts[countsPeakIndex+halfinterval]
    return leftParameter, rightParameter, peakLocation
