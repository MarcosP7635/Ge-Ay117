import numpy as np
import math
import matplotlib as plt
import matplotlib.pyplot as plt
from mpmath import*
import os
filetorun = os.getcwd() + "\Set1\Set1problem1a.py"
exec(open(filetorun).read())
def calculateX (alpha, beta, smalln, bign):
    x = [0]*smalln
    feederX = [0]*bign
    probs = [0]*bign
    counts = np.empty(0)
    '''
    will randomly choose integers (0 to 100)
    will compute their probabilities
    '''
    for i in range(bign):
        feederX[i] = np.random.uniform(0, 100)
        probs[i] = beta/(pi*(beta**2+(feederX[i]**2-alpha**2)))
        count = [feederX[i]]*int(probs[i]*(bign))
        counts = np.append(counts, count)
#now I need to draw from this distribution.
    for i in range(smalln):
        index = int(np.random.uniform(0, len(counts)-1))
        x[i] = counts[index]
    return x
#now to use kaden's data
kadenData = np.array([2.1501584946622896, 2.059141036875201, 3.202064945118297, 0.3185223933143999, 2.7247008353490276, 2.1872906588131653, 3.531793402803334, 2.6268228279674313, 1.7750725822474034, 2.455717658454668, 4.082327116188983, 3.517720739771047, 1.559523295323002, 3.3846486574981722, 4.105044811471006, 3.503691217529488])
#now I need to add the data 1 at a time.
#sigma = np.array([1])
sigma = np.empty(0)
#fig, axs = plt.subplots(0)
#for i in range(1,len(kadenData)+1):
i = 17
data = kadenData[:i+1]
mu = np.average(data)
sigma = np.append(sigma, i**-.5)
num=i
print(num)
#main(iterations, Ai, sigma, gaussians, w)
main(10**3, data, sigma, len(data), num)
#plt.tight_layout()
#fig.savefig(os.getcwd()+'\Set 2\plot' + str(i) + '.png')
#for ax in axs.flat:
