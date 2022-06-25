'''
--------------------------------------------------------------------------------------------------
Assignment 4 - EE2703 (Jan-May 2022)
Purvam Jain (EE20B101)
--------------------------------------------------------------------------------------------------
'''

# Let's start by importing necessary libraries and constants

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.linalg import lstsq
from sklearn.metrics import mean_squared_error

pi = np.pi 
e = np.e

# ***********************************QUESTION 1***************************************
''' We just have to define functions and plot them.'''
# Start by defining functions e^x and cos(cos(x)) as stated in the question

def f(x):
    return np.exp(x)

def g(x):
    return np.cos(np.cos(x))

# Define the points to plot the functions 
x = np.linspace(-2*pi,4*pi,500) # Generates array of 500 points in given range [-2pi,4pi]

plt.grid()
plt.plot(x,f(x))
plt.title('Plot of f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()

# Since the exponential function increases rapidly we plot it in semilog to get a linear visualization
plt.grid()
plt.semilogy(x,f(x))
plt.title('Plot of exp(x) in semilog scale')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()

plt.grid()
plt.plot(x,g(x))
plt.title('Plot of cos(cos(x))')
plt.xlabel('x')
plt.ylabel('g(x)')
plt.show()

# ***********************************QUESTION 2***************************************
''' Obtain the first 51 Fourier coefficients for the two functions above. '''

def fourier_coeff_calc(f,n):
    ''' f will be input function that is f(x) or g(x)
        This function calculates first n Fourier coefficients
        of given input function f. '''
    
    
    # Start by defining functions u(x,k) and v(x,k) as stated in the question
    u = lambda x,k : f(x)*np.cos(k*x)
    v = lambda x,k : f(x)*np.sin(k*x)
    
    # Create empty arrays
    f_c = np.empty(n) # To store fourier cofficients for f(x)
    # for k in range(n):
    #     if k%2==0:
    #         f_c[k],_ = integrate.quad(u,0,2*pi,args=(k/2))
    #     else:
    #         f_c[k],_ = integrate.quad(v,0,2*pi,args=((k+1)/2))
    f_c[0] = integrate.quad(f,0,2*np.pi)[0]
    for i in range(1,n):
        if(i%2==1):
            f_c[i] = integrate.quad(u,0,2*np.pi,args=(int(i/2)+1))[0]
        else:
            f_c[i] = integrate.quad(v,0,2*np.pi,args=(int(i/2)))[0]
    
        
    f_c /= pi
    f_c[0] /= 2
    return f_c

coeff_exp = fourier_coeff_calc(f,51)
coeff_cos = fourier_coeff_calc(g,51)

# ***********************************QUESTION 3***************************************
'''For each of the two functions, make two different plots using “semilogy” and “loglog” 
and plot the magnitude of the coefficients vs n.'''

def coeff_plotter(coeffs,fname):
    plt.grid()
    plt.semilogy(abs(coeffs),'o',color = 'r',markersize = 4)
    plt.title('Fourier Coefficients for {} by direct integration (semilog)'.format(fname))
    plt.xlabel('n')
    plt.ylabel('Fourier Coefficients') 
    plt.show()


    plt.grid()
    plt.loglog(abs(coeffs),'o',color = 'r',markersize = 4)
    plt.title('Fourier Coefficients for {} by direct integration (loglog)'.format(fname))
    plt.xlabel('n')
    plt.ylabel('Fourier Coefficients') 
    plt.show()

coeff_plotter(coeff_exp,'e^x')
coeff_plotter(coeff_cos,'cos(cos(x))')

# ***********************************QUESTION 4***************************************
'''We instead do a “Least Squares approach” to the problem.'''


def linearsquares(func,n):
    X=np.linspace(0,2*pi,401)
    X=X[:-1]    # drop last term to have a proper periodic integral
    b=func(X)  # f has been written to take a vector
    A=np.zeros((400,n))    # allocate space for A
    A[:,0]=1    # col 1 is all ones
    for k in range(1,(n+1)//2):
        A[:,2*k-1]=np.cos(k*X)  # cos(kx) column
        A[:,2*k]=np.sin(k*X)    # sin(kx) column
    #endfor
    c1=lstsq(A,b)[0]    # the ’[0]’ is to pull out the best fit vector. 
    # lstsq returns a list of Fourirer Coefficients.
    return c1,A,X

c,A,X = linearsquares(f,51)
d,A,X = linearsquares(g,51)

# ***********************************QUESTION 5***************************************
'''Obtain the coefficients for both the given functions. Plot them with green circles in
the corresponding plots.'''

def lstsq_plotter(coeffs,fname):
    plt.grid()
    plt.plot(coeffs,'o',color = 'g')
    plt.title('Fourier Coefficients calculated using lstsq method for {}'.format(fname))
    plt.xlabel('n')
    plt.ylabel('Coefficients')
    plt.show()

lstsq_plotter(c,'exp(x)')
lstsq_plotter(d,'cos(cos(x)')

# ***********************************QUESTION 6***************************************
''' Compare the answers got by least squares and by the direct integration. How much 
deviation is there (find the absolute difference between the two sets of coefficients 
and find the largest deviation. How will you do this using vectors?)'''

print("Largest Deviation in exp(x): ", max(abs(c-coeff_exp)))
print("Largest Deviation in cos(cos(x)): ", max(abs(d-coeff_cos)))

# ***********************************QUESTION 7***************************************
''' Compute Ac from the estimated values of c. These should be the function values at xi.
Plot them(with green circles) in Figures 1 and 2 respectively for the two functions.'''

def plotter(coeffs,func):
    lstsqresults = A.dot(coeffs)
    plt.grid()
    plt.plot(X,lstsqresults,'o',color = 'g', markersize = 2)
    plt.plot(X,func(X))
    plt.title(' Original Function and Estimated Function')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend(['Original','Estimated'])
    plt.show()

plotter(c,f)
plotter(d,g)
