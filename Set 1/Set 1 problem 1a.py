import numpy as np
import math
import matplotlib as plt
import matplotlib.pyplot as plt
from mpmath import*
#mpmath is essential for performing these calculations on small probabilities
#mpmath also has a built-in very accurate value for e
def calculateP (Ai, mu, sigma):
    p = e**((-.5*(((Ai-mu)/sigma)**2)))
    p = p*((1/(sigma*((2*pi)**.5))))
    #p = (p+760)/-300
#This is the probability of obtaining a specific value of Ai given mu and sigma
    return p
def makePlot(iterations, Ai, sigma):
    muDistr = np.random.uniform(Ai+3*sigma,Ai-3*sigma, iterations)
    AiArray = [0]*iterations
    #Calculate the unnormalized distribution
    for i in range(len(muDistr)):
        mu = muDistr[i]
        AiArray[i] = calculateP(Ai, mu, sigma)
        #print(AiArray)
    min = np.amin(AiArray)
    max = np.amax(AiArray)
    #Now to normalize the array
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
    p = e**((-.5*(((Ai-mu)/sigma)**2)))
    p = p*((1/(sigma*((2*pi)**.5))))
    p = (p-min)/(max-min)
    return p
A1 = 41.4
A2 = 46.9
#Value preset by the problem
#call everything make the plot
makePlot(10**5,A1,2)
#makePlot(10**5,A2,3)
