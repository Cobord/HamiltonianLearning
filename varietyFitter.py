import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

import scipy
import numpy as np
from scipy import linalg, matrix

def lowestEigenspaces(A,num=1):
    u, s, vt = scipy.linalg.svd(A)
    n = len(s)
    # reverse the n first columns of u
    u[:,:n] = u[:, n-1::-1]
    # reverse s
    s = s[::-1]
    # reverse the n first rows of vt
    vt[:n, :] = vt[n-1::-1, :]
    #print(vt)
    #print(vt[0:num])
    #print(vt[0:num,:])
    null_space = vt[0:num]
    return scipy.transpose(null_space)

def null(A,eps=1e-9):
    u, s, vh = scipy.linalg.svd(A)
    padding = max(0,np.shape(A)[1]-np.shape(s)[0])
    null_mask = np.concatenate(((s <= eps), np.ones((padding,),dtype=bool)),axis=0)
    null_space = scipy.compress(null_mask, vh, axis=0)
    return scipy.transpose(null_space)

dataPoints = np.array([[1.0,0],
                        [3.0/5,4.0/5],
                        [4.0/5,3.0/5],
                        [-12.0/13,5.0/13.0],
                        [-5.0/13,-12.0/13.0],
                        [0,-1],
                        ])

#noise the data so no longer exactly a circle
dataPoints = dataPoints + .05*np.random.randn(6,2)

plt.scatter(dataPoints[:,0],dataPoints[:,1])
plt.gca().set_aspect('equal')
plt.savefig('circleTestCase.png')
plt.show()

poly= PolynomialFeatures(degree=2)
monomials=poly.fit_transform(dataPoints)
#print(monomials)

# the .1 here is chosen to make the threshold instead of exact null space
#defining_equation=null(monomials,.1)
defining_equation=lowestEigenspaces(monomials)
print(defining_equation)
# it should be -1*1+0*x+0*y+1*x^2+0*xy+1*y^2 or something proportional

#TODO: do same for other plane curves of higher degree to see how well it scales
# in planeCurves.py