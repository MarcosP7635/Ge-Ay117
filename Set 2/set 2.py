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
