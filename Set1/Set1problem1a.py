import numpy as np
import math
import matplotlib as plt
import matplotlib.pyplot as plt
from mpmath import*
import pandas as pd
import os
filetorun = os.getcwd() + "\Set 2\confidence interval set2.py"
exec(open(filetorun).read())
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
def main(iterations, Ai, sigma, gaussians, num):
    if (gaussians<2):
        if isinstance(sigma,(list,pd.core.series.Series,np.ndarray)):
            sigma = sigma[0]
        muDistr = np.arange(-5,5, 10/iterations)
        #muDistr = np.arange(Ai-3*sigma,Ai+3*sigma, 6*sigma/iterations)
        AiArray = [0]*len(muDistr)
        #Calculate the unnormalized distribution
        for i in range(len(muDistr)):
            mu = muDistr[i]
            AiArray[i] = calculateP(Ai, mu, sigma)
            #print(AiArray)
            #Now to normalize the array
    else:
        mu = calculateMu(Ai, sigma)
        sigma = sigma[len(sigma)-1]
        #sigma = calculateSigma(sigma)
        print(mu, sigma)
        main(iterations, mu, sigma, 1, num)
        return
        #The code below is technically wrong but I don't want to delete it
        '''min = np.amin(Ai)-3*sigma
        max = np.amax(Ai)+3*sigma
        steps = (max-min)/iterations
        muDistr = np.arange(min,max,steps)
        AiArray = [1]*(len(muDistr))
        #Calculate the unnormalized distribution
        for i in range(len(muDistr)):
            mu = muDistr[i]
            for j in range(len(Ai)):
                AiArray[i] = AiArray[i]*calculateP(Ai[j], mu, sigma)'''
        #Now to normalize the array
    AiArray = normalize(AiArray,muDistr)
    makePlot(muDistr, AiArray, Ai, num)
    print(ConfidenceInterval(muDistr, AiArray, .68))
    print(ConfidenceInterval(muDistr, AiArray, .95))
'''
Normalizes the y values such that when integrated with respect to the x values
we get 1
'''
def normalize(AiArray, muDistr):
    muDistr = muDistr.flatten()
    print(len(AiArray),muDistr.shape)
    normConstant = np.trapz(AiArray, muDistr)
    for i in range(len(muDistr)):
       AiArray[i] = AiArray[i]/normConstant
    print("normConstant = " + str(normConstant))
    return AiArray
#makes the plot given x array, y array, and Ai
def makePlot(muDistr, AiArray, Ai, num):
    #plt.scatter(muDistr,AiArray)
    if(num==17):
        plt.scatter(muDistr,AiArray)
        title = "P(alpha|x,I) "
        plt.ylabel("Normalized Probability")
        plt.xlabel('x')
        plt.title(title)
        plt.show()

'''
ax1 = fig.add_subplot(4, 4, num)
ax1.plot(muDistr,AiArray)
#plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
#wpsace = 1.0, hspace = 1.0)
ax1.set_title('Using ' + str(num) + ' Data Points', size = 8)
ax1.set_ylabel("P(alpha|X,I)", size = 8)
ax1.set_xlabel('mu', size = 8)
'''
#Value preset by the problem
def calculateSigma(sigmaArray):
    newSigma = sigmaArray[0]
    for i in range(len(sigmaArray)-1):
        newSigma = (newSigma*sigmaArray[i+1])/sqrt(newSigma**2+sigmaArray[i+1]**2)
    return newSigma
def calculateMu(AiArray, sigmaArray):
    mu = AiArray[0]
    sigma = sigmaArray[0]
    for i in range(len(sigmaArray)-1):
        mu = ((sigmaArray[i+1]**2)*mu)+(AiArray[i+1]*(sigma**2))
        mu = mu/(sigma**2+sigmaArray[i+1]**2)
        sigma = calculateSigma([sigma,sigmaArray[i+1]])
    return mu
#call everything make the plot
#the fourth input (gaussians) should equal the length of the array named Ai
#if you only wish to plot a single gaussian just put gaussian as 1 and enter an
#float for Ai
#main(10**5,A1,2,1)
#main(10**5,A2,3,1)
'''
A1 = 41.4
A2 = 46.9
A3 = 44.1
mu = (9*A1+4*A2)/13
sigma2 = calculateSigma([2,3])
sigma3 = calculateSigma([2,3,6.1])
print(sigma3)
mu = ((sigma2**2)*mu+A3*(6.1**2))/(6.1**2+sigma2**2)
print("mu = "+ str(mu))
prob1 = calculateP(41.4,mu,sigma3)/ 4.01335922875334
prob2 = calculateP(46.9, mu,sigma3)/ 4.01335922875334
prob3 = calculateP(44.1,mu,sigma3)/ 4.01335922875334
print(prob1*prob2*prob3)
mu = [38.9, 34.2, 52.1, 38.7, 44.1, 40.4, 48.7, 48.0, 38.1, 35.7 ]
sigma = [6.8, 7.4, 6.3, 5.0, 6.6, 5.4, 5., 5.9, 5., 6.9]
FiveA = 1
for i in range(len(mu)):
    FiveA = FiveA*calculateP(mu[i], 43.16210724799415, 1.6054330245612716)/4.01335922875334
print(FiveA)
main(10**5,[A1,A2,A3],[2,3,6.1],3)
'''
print("hello")
