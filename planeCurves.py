import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

def myCircle(x,y):
    return x**2+y**2-1

def myCircle2(x):
    return [x[0]**2+x[1]**2-1,0]

def ellipticCurve(x,y):
    a = 3
    b = 2
    return y**2-x**3-a*x-b
    
def ellipticCurve2(x):
    a = 3
    b = 2
    return [x[1]**2-x[0]**3-a*x[0]-b,0]

def selectDataPointsPlaneCurve2(equation):
    x = np.linspace(-3, 3, 10)
    y = np.linspace(-3, 3, 10)
    z = np.zeros((10*10,2))
    i=0
    for x0 in x:
        for y0 in y:
            z0 = scipy.optimize.root(equation,[x0,y0]).x
            z[i]=z0
            i=i+1
    return z

def selectDataPointsPlaneCurve(equation):
    x = np.linspace(-3, 3, 10)
    y = np.linspace(-3, 3, 10)
    x,y = np.meshgrid(x, y)
    z = equation(x,y)
    CS = plt.contour(x,y,z,[0.0])
    dat0= CS.allsegs[0][0]
    return dat0
    
dataPoints=selectDataPointsPlaneCurve2(ellipticCurve2)
#dataPoints=np.unique(dataPoints,axis=0) depends on numpy version
plt.scatter(dataPoints[:,0],dataPoints[:,1])
plt.title("Elliptic Curve y^2 = x^3 + {0}*x+{1}".format(3,2))
plt.show()
plt.close()

dataPoints=selectDataPointsPlaneCurve2(myCircle2)
#dataPoints=np.unique(dataPoints,axis=0)
plt.scatter(dataPoints[:,0],dataPoints[:,1])
plt.title("Unit Circle")
plt.show()
plt.close()