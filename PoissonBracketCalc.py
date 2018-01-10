import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

import scipy
import numpy as np
from scipy import linalg, matrix

degFreedom = 3
numInputs = 2*degFreedom
degree=2

dataPoint=np.zeros(numInputs)
dataPoint=dataPoint.reshape(1,numInputs)

poly= PolynomialFeatures(degree)
monomials=poly.fit_transform(dataPoint)
lengthSummands=monomials.size

equation1 = np.random.randn(lengthSummands)
equation1=equation1.reshape(1,lengthSummands)
print(equation1)

equation2 = np.random.randn(lengthSummands)
equation2=equation2.reshape(1,lengthSummands)
print(equation2)

class Memoize:
    def __init__(self, fn):
        self.fn = fn
        self.memo = {}
    def __call__(self, *args):
        if args not in self.memo:
    	    self.memo[args] = self.fn(*args)
        return self.memo[args]

#@Memoize
def PoisHelper(monomial1,monomial2,numInputs,lengthSummands):
    returnFunction=np.zeros(lengthSummands)
    totalDegree1=np.sum(monomial1)
    totalDegree2=np.sum(monomial2)
    if (totalDegree1 == 0 || totalDegree2 == 0):
        return returnFunction
    elif (totalDegree1 == 1 && totalDegree2 == 1):
        

def PoisB(equation1,equation2,numInputs,poly):
    lengthSummands=equation1.size
    returnFunction=np.zeros_like(equation1)
    for i in range(lengthSummands):
        eq1Coeff = equation1[:,i][0]
        eq1PowerList = poly.powers_[i,range(numInputs)]
        for j in range(lengthSummands):
            eq2Coeff = equation1[:,i][0]
            eq2PowerList = poly.powers_[i,range(numInputs)]
            returnFunction=returnFunction+eq1Coeff*eq2Coeff*PoisHelper(eq1PowerList,eq2PowerList,numInputs,lengthSummands)
    return -1