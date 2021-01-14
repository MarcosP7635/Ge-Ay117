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
Takes the number of x values (iterations), Ai, and the standard deviation
to call helper functions to calculate the array of x and y values
then calls a final function to produce plots
'''
def main(iterations, Ai, sigma, gaussians):
    AiArray = [0]*iterations
    if (gaussians<2):
        muDistr = np.arange(Ai-3*sigma,Ai+3*sigma, 6*sigma/iterations)
        #Calculate the unnormalized distribution
        for i in range(len(muDistr)):
            mu = muDistr[i]
            AiArray[i] = calculateP(Ai, mu, sigma)
            #print(AiArray)
            #Now to normalize the array
    else:
        min = np.amin(Ai)-3*sigma
        max = np.amax(Ai)+3*sigma
        steps = (max-min)/iterations
        muDistr = np.arange(min,max,steps)
        AiArray = [0]*(len(muDistr))
        #Calculate the unnormalized distribution
        for i in range(len(muDistr)):
            intermediateP = 1
            mu = muDistr[i]
            for j in range(len(Ai)):
                intermediateP = intermediateP*calculateP(Ai[j], mu, sigma)
            AiArray[i] = intermediateP
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
#Value preset by the problem
A1 = 41.4
A2 = 46.9
def calculateSigma(sigmaArray):
    newSigma = sigmaArray[0]
    for i in range(len(sigmaArray)-1):
        newSigma = (newSigma*sigmaArray[i+1])/math.sqrt(newSigma**2+sigmaArray[i+1]**2)
    return newSigma
#call everything make the plot
#the fourth input (gaussians) should equal the length of the array named Ai
#if you only wish to plot a single gaussian just put gaussian as 1 and enter an
#float for Ai
#main(10**5,A1,2,1)
#main(10**5,A2,3,1)
main(10**6,[A1,A2],calculateSigma([2,3]),2)
