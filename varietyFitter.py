import matplotlib.pyplot as plt
#import tensorflow as tf
from sklearn.preprocessing import PolynomialFeatures

import scipy
import numpy as np
from scipy import linalg, matrix

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

plt.scatter(dataPoints[:,0],dataPoints[:,1])
plt.gca().set_aspect('equal')
plt.savefig('circleTestCase.png')
plt.show()

poly= PolynomialFeatures(degree=2)
monomials=poly.fit_transform(dataPoints)
#print(monomials)

defining_equation=null(monomials)
print(defining_equation)
# it should be -1*1+0*x+0*y+1*x^2+0*xy+1*y^2 or something proportional

#TODO: do same for other plane curves of higher degree to see how well it scales
#TODO: noise the data so no longer exactly null space, idea randomly select subsets