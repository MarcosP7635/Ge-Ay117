import numpy as np
import math
import matplotlib as plt
import matplotlib.pyplot as plt
from mpmath import*
#mpmath is essential for performing these calculations on small probabilities
#mpmath also has a built-in very accurate value for e
def calculateP (Ai, mu, sigma):
    p = e**((-.5*(((Ai-mu)/sigma)**2)))
    #p = p*((1/(sigma*((2*pi)**.5))))
    #Using the above line will normalize the data
#Returns the probability of obtaining a specific value of Ai given mu and sigma
    return p
'''
Main takes the number of x values (iterations), Ai, and the standard deviation
to call helper functions to calculate the array of x and y values
then calls a final function to produce plots
'''
def plotSingularGaussian(iterations, Ai, sigma):
    muDistr = np.arange(Ai-3*sigma,Ai+3*sigma, 6*sigma/iterations)
    AiArray = [0]*iterations
    #Calculate the unnormalized distribution
    for i in range(len(muDistr)):
        mu = muDistr[i]
        AiArray[i] = calculateP(Ai, mu, sigma)
        #print(AiArray)
        #Now to normalize the array
    AiArray = normalize(AiArray,muDistr)
    makePlot(muDistr, AiArray, Ai)
'''
Normalizes the y values such that when integrated with respect to the x values
we get 1
'''
def normalize(AiArray, muDistr):
    normConstant = np.trapz(AiArray, muDistr)
    print(normConstant)
    for i in range(len(muDistr)):
       AiArray[i] = AiArray[i]/normConstant
    return AiArray
#makes the plot given x array, y array, and Ai
def makePlot(muDistr, AiArray, Ai):
    plt.scatter(muDistr,AiArray)
    title = "Normalized Probability of obtaining Ai = " + str(Ai)
    plt.ylabel("Normalized Probability")
    plt.xlabel('mu')
    plt.title(title)
    plt.show()
#This function calculates the normalized probability
A1 = 41.4
A2 = 46.9
#Value preset by the problem
#call everything make the plot
plotSingularGaussian(10**5,A1,2)
plotSingularGaussian(10**5,A2,3)
