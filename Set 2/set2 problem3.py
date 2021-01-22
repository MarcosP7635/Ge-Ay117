import numpy as np
import math
import matplotlib as plt
import matplotlib.pyplot as plt
from mpmath import*
import os
filetorun = os.getcwd() + "\Set 2\confidence interval set2.py"
exec(open(filetorun).read())
file1 = os.getcwd() + '\Set 2\ps2_posterior_05.txt'
file2 = os.getcwd() + '\Set 2\ps2_posterior_10.txt'
file3 = os.getcwd() + '\Set 2\ps2_posterior_20-1.txt'
file4 = os.getcwd() + '\Set 2\ps2_posterior_50.txt'
fileArray = [file1,file2,file3,file4]
print(fileArray)
convertedFileArray = [None]*4
for f in range(len(fileArray)):
    convertedFileArray[f] = np.loadtxt(fileArray[f])
    actingArray = convertedFileArray[f]
    print(ConfidenceInterval(actingArray[0:,0],actingArray[0:,1], .68))
    print(ConfidenceInterval(actingArray[0:,0],actingArray[0:,1], .95))
#print(convertedFileArray)
