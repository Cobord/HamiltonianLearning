import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from math import *
import sympy as sy

def harmonic_oscillator_deriv(x,t):
    omega = 3.0
    mass = .5
    # H = p^2 + 1/2 .5 3.0^2 x^2
    # H = p^2 + 2.25 x^2
    nx0 = x[0]
    npx0 = x[1]
    res = np.array([npx0,-mass*omega**2*nx0])
    return res

#def pendulum_deriv(x,t):
#    grav = 9.8
#    mass = .5
#    length = 1
    # H = p^2 / 2 (mass * length^2) - mass * grav * length * sin theta
    
def van_der_pol_oscillator_deriv(x,t):
    mu = 3.0
    # H = px py + x y - 3.0 (1-x^2) y py
    # H = px py + x y - 3.0 y py + 3.0 x^2 y py
    nx0=x[0]
    ny0=x[1]
    npx0=x[2]
    npy0=x[3]
    res=np.array([npy0,npx0-mu*(1-nx0**2)*ny0,-ny0-2*mu*nx0*ny0*npy0,-nx0+mu*npy0-mu*nx0**2*npy0])
    #if(t==0):
    #    print(res)
    #    print(type(res))
    #    print(res[0])
    #    print(type(res[0]))
    return res

def volterra_deriv(x,t):
    epsilon_1 = -.2
    epsilon_2 = 0
    a_12 = 4.0
    a_21 = -a_12
    # H = -.2 q1 + 0 q2 - e^(p1 + 1/2 4 q2) - e^(p2 + 1/2 -4 q1)
    # H = -.2 q1 + e^p1 (e^(q2))^2 - e^p2 (e^q1)^(-2)
    
    q1, q2, p1, p2 = sy.symbols('q1 q2 p1 p2', real=True)
    H = epsilon_1*q1 + epsilon_2*q2 - sy.exp(p1+.5*a_12*q2) - sy.exp(p2+.5*a_21*q1)
    
    dotq1=-sy.diff(H,p1)
    dotq2=-sy.diff(H,p2)
    dotp1=sy.diff(H,q1)
    dotp2=sy.diff(H,q2)
    
    nq1 = x[0]
    nq2 = x[1]
    npq1 = x[2]
    npq2 = x[3]
    
    ndotq1=dotq1.subs([(q1,nq1),(q2,nq2),(p1,npq1),(p2,npq2)]).evalf()
    ndotq2=dotq2.subs([(q1,nq1),(q2,nq2),(p1,npq1),(p2,npq2)]).evalf()
    ndotp1=dotp1.subs([(q1,nq1),(q2,nq2),(p1,npq1),(p2,npq2)]).evalf()
    ndotp2=dotp2.subs([(q1,nq1),(q2,nq2),(p1,npq1),(p2,npq2)]).evalf()
    ndotq1=np.float64(ndotq1)
    ndotq2=np.float64(ndotq2)
    ndotp1=np.float64(ndotp1)
    ndotp2=np.float64(ndotp2)
    
    res = np.array([ndotq1,ndotq2,ndotp1,ndotp2])
    #print(res)
    #print(type(res))
    #print(res[0])
    #print(type(res[0]))
    return res

#def kuramoto_deriv(x,t):

#def double_pendulum_deriv(x,t):

#def coupled_osc_deriv(x,t):

ts = np.linspace(0.0, 50.0, 500)
x0 = [4.0, 0.0]
xs = odeint(harmonic_oscillator_deriv,x0,ts)
plt.plot(ts,xs[:,0])
plt.savefig('harmonic_oscillator_time.png')
plt.show()

ts = np.linspace(0.0,50.0,500)
x0 = [4.0, 0.0]
xs = odeint(harmonic_oscillator_deriv,x0,ts)
x = xs[:,0]
p = xs[:,1]
inputdata=np.dstack((x,p,ts))[0]
print(inputdata)

x0 = [0.0, 2.0]
xs = odeint(harmonic_oscillator_deriv,x0,ts)
x = xs[:,0]
p = xs[:,1]
inputdata=np.vstack((inputdata,np.dstack((x,p,ts))[0]))
print(inputdata)

x0 = [1.0, 2.0]
xs = odeint(harmonic_oscillator_deriv,x0,ts)
x = xs[:,0]
p = xs[:,1]
inputdata=np.vstack((inputdata,np.dstack((x,p,ts))[0]))
print(inputdata)

ts = np.linspace(0.0, 50.0, 500)
x0 = [4.0, 0.0,0.0,4.0]
xs = odeint(van_der_pol_oscillator_deriv,x0,ts)
plt.plot(ts,xs[:,0])
plt.savefig('vanderpol_oscillator_time.png')
plt.show()

ts = np.linspace(0.0, 50.0, 500)

x0 = [0.0, 0.0,.4,.9]
xs = odeint(volterra_deriv,x0,ts)
plt.plot(ts,xs[:,0])
plt.savefig('volterra_time.png')
plt.show()