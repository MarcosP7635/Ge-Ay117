import numpy as np
import math
import matplotlib as plt
import matplotlib.pyplot as plt
def calculateP (Ai, mu, sigma):
    #sigma = 2
#I think the problem specifies this value but I will have to check with someone later
    p = (((np.log((1/(sigma*math.sqrt(2*np.pi))))+((-.5*((Ai-mu/sigma)**2)))))+987.2907100867332)/(987.2907100867332-738.8943667164641)
    #p = (np.e**((-.5*((Ai-mu/sigma)**2)+987.2907100867332)))
    #p = p*((1/(sigma*math.sqrt(2*np.pi))))
    #p = (p+760)/-300
#This is the probability of obtaining a specific value of Ai given mu and sigma
    return p
print(calculateP(1, 1, 2))
A1 = 41.4
def makePlot(iterations, Ai, sigma):
    muDistr = np.random.uniform(-3*sigma,3*sigma, iterations)
    AiArray = [0]*iterations
    for i in range(len(muDistr)):
        mu = muDistr[i]
        AiArray[i] = calculateP(Ai, mu, sigma)
        #print(AiArray)
    print(np.amin(AiArray))
    print(np.amax(AiArray))
    plt.scatter(muDistr,AiArray)
    #plt.title('Probability of obtaining Ai = ', Ai
    yaxis = "Normalized natural log of the Probability of obtaining Ai = " + str(Ai)
    plt.ylabel(yaxis)
    plt.xlabel('mu')
    plt.show()
makePlot(10**5,A1,2)
