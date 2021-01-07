import numpy as np
import math
import matplotlib as plt
import matplotlib.pyplot as plt
from mpmath import*
def calculateP (Ai, mu, sigma):
    p = e**((-.5*((Ai-mu/sigma)**2)))
    p = p*((1/(sigma*((2*pi)**.5))))
    #p = (p+760)/-300
#This is the probability of obtaining a specific value of Ai given mu and sigma
    return p
print(calculateP(1, 1, 2))
A1 = 41.4
#Value preset by the problem
def makePlot(iterations, Ai, sigma):
    muDistr = np.random.uniform(2*Ai+3*sigma,2*Ai-3*sigma, iterations)
    #The actual peak for the probability is between 75 and 90
    AiArray = [0]*iterations
    for i in range(len(muDistr)):
        mu = muDistr[i]
        AiArray[i] = calculateP(Ai, mu, sigma)
        #print(AiArray)
    min = np.amin(AiArray)
    max = np.amax(AiArray)
    plt.scatter(muDistr,AiArray)
    yaxis = "Probability of obtaining Ai = " + str(Ai)
    plt.ylabel(yaxis)
    plt.xlabel('mu')
    plt.show()
    for i in range(len(muDistr)):
        mu = muDistr[i]
        AiArray[i] = normalizeP(max,min, Ai, mu, sigma)
    plt.scatter(muDistr,AiArray)
    yaxis = "Normalized Probability of obtaining Ai = " + str(Ai)
    plt.ylabel(yaxis)
    plt.xlabel('mu')
    plt.show()
#This function calculates the normalized probability
def normalizeP (max, min, Ai, mu, sigma):
    p = e**((-.5*((Ai-mu/sigma)**2)))
    p = p*((1/(sigma*((2*pi)**.5))))
    p = (p-min)/(max-min)
    return p
#call everything make the plot
makePlot(10**5,A1,2)
