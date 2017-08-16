import matplotlib.pyplot as plt
#import tensorflow as tf
from sklearn.preprocessing import PolynomialFeatures

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
print(monomials)

#TODO: still have to calculate nullspace of the monomials matrix
#that gives the quadratic that best fits this variety
# it should be -1*1+0*x+0*y+1*x^2+0*xy+1*y^2
# copy nullspace code from https://stackoverflow.com/questions/5889142/python-numpy-scipy-finding-the-null-space-of-a-matrix

#TODO: do same for other plane curves of higher degree to see how well it scales
#TODO: noise the data so no longer exactly null space