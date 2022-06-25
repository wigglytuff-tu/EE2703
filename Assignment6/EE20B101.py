'''
--------------------------------------------------------------------------------------------------
Assignment 6 - EE2703 (Jan-May 2022)
Purvam Jain (EE20B101)
--------------------------------------------------------------------------------------------------
'''

import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt

'''
Question1: Solve for the time response of a spring satisfying
        x"+2.25x = f(t)
with x(0) = 0 and ˙x = 0 for t going from zero to 50 seconds. Use system.impulse to
do the computation'''

# Defining transfer function F(s)
def transfer_func(a,decay):
    pol = np.polymul([1.0,0,2.25] , [1,-2*decay, a**2 + decay**2])
    return sp.lti([1,-decay],pol)

t,x = sp.impulse(transfer_func(1.5,-0.5),None,np.linspace(0,50,1001))   # To get Inverse Laplace Transform we compute step response

plt.plot(t,x)
plt.title("Time response of spring with decay=0.5")
plt.xlabel("Time")
plt.ylabel("x")
plt.show()

'''Question 2: Repeat Question 1 with decay=0.05'''

t,x = sp.impulse(transfer_func(1.5,-0.05),None,np.linspace(0,50,1001))

plt.plot(t,x)
plt.title("Time response of spring with decay=0.05")
plt.xlabel("Time")
plt.ylabel("x")
plt.show()

'''
Question 3: Obtain the system transfer function X(s)/F(s). Now use signal.lsim to simulate the
problem. In a for loop, vary the frequency of the cosine in f(t) from 1.4 to 1.6 in
steps of 0.05 keeping the exponent as exp(-0.05t) and plot the resulting responses.
'''



for f in np.arange(1.4,1.6,0.05):
    TF = sp.lti([1],[1,0,2.25])     # Defining transfer function X(s)/F(s)
    t = np.linspace(0,200,1001)
    fx = np.cos(f*t)*np.exp(-0.05*t)
    t,x,_ = sp.lsim(TF,fx,t)            # Convolution of X(s)/F(s) with f(t)
    plt.plot(t,x)
    plt.title("Forced Damping Oscillator with frequency: "+str(f))
    plt.xlabel("Time")
    plt.ylabel("x")
    plt.show()

'''
Question 4:Solve for a coupled spring problem:
            x"+ (x-y) = 0
            y"+2(y-x) = 0
where the initial condition is x(0) = 1, ˙x(0) = y(0) = y˙(0) = 0. Substitute for y from
the first equation into the second and get a fourth order equation. Solve for its time
evolution, and from it obtain x(t) and y(t) for 0 ≤ t ≤ 20.'''

Xs = sp.lti([1,0,2],[1,0,3,0])          # Defining X(s)
t,x = sp.impulse(Xs,None,np.linspace(0,20,1001))    # Computing ILT

Ys = sp.lti([2],[1,0,3,0])          # Defining Y(s)
t,y = sp.impulse(Ys,None,np.linspace(0,20,1001))    # Computing ILT

plt.plot(t,x)
plt.plot(t,y)
plt.title("Coupled Oscillations of spring")
plt.xlabel("Time")
plt.legend(['x','y'])
plt.show()

'''
Question 5: . Obtain the magnitude and phase response of the Steady State Transfer function of
a two-port network.'''

H = sp.lti([1],[1e-12,1e-3,1])      # Defining steady state transfer function for LCR
w,S,phi=H.bode()

plt.semilogx(w,S)
plt.xlabel("")
plt.ylabel("Magnitude")
plt.title("Magnitude Response")
plt.show()

plt.semilogx(w,phi)
plt.xlabel("log(w)")
plt.ylabel("Phase")
plt.title("Phase Response")
plt.show()

'''
Question 6:' Repeat Question 5 supposing the input signal Vi(t) is given by
                                Vi(t) = cos(10^3t)u(t) - cos(10^6t)u(t)

Obtain the output voltage v0(t) by defining the transfer function as a system and
obtaining the output using signal.lsim '''

t = np.arange(0,30e-6,10e-8)
Vi = np.cos(1e3*t) - np.cos(1e6*t)
t,y,svec = sp.lsim(H,Vi,t)
plt.plot(t,y)
plt.xlabel('Time')
plt.ylabel('V_output')
plt.title('Output signal in micro-seconds scale')
plt.show()

tms = np.arange(0,30e-3,10e-6)
Vims = np.cos(1e3*tms) - np.cos(1e6*tms)
tms,yms,svecms = sp.lsim(H,Vims,tms)
plt.plot(tms,yms)
plt.xlabel('Time')
plt.ylabel('V_output')
plt.title('Output signal in milli-seconds scale')
plt.show()

