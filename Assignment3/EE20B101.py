'''
--------------------------------------------------------------------------------------------------
Assignment 3 - EE2703 (Jan-May 2022)
Purvam Jain (EE20B101)
--------------------------------------------------------------------------------------------------
'''
import pylab
import argparse
import numpy as np
import scipy.special as sp
from scipy.linalg import lstsq
from sklearn.metrics import mean_squared_error


parser = argparse.ArgumentParser('file',description='Optional data file from user.', argument_default="fitting.dat")
parser.add_argument('file', type=argparse.FileType('r'), nargs='?', default='fitting.dat',
help="Optional input for your own data file")
args = parser.parse_args()

data = pylab.loadtxt(args.file)

# print(data.dtype)
# print(data.shape)

def g(t, A, B):		# Computes and returns the function A*J(2, t) + B*t
	y = A*sp.jn(2, t) + B*t
	return y
N,k,true_A,true_B = 101,9,1.05,-0.105    #Declare used constants
sigma=np.logspace(-1,-3,9)
t = data[:,0]                     # Extract time values from zeroth column 
yy = data[:,1:]                   # Extract function outputs

# ***********************************QUESTION 4***************************************

pylab.plot(t,yy)
pylab.plot(t,g(t,true_A,true_B),color='black')
pylab.xlabel(r'$t$',size=20)
pylab.ylabel(r'$f(t)+noise$',size=20)
pylab.title(r'Plot of the data to be fitted')
legends = list(np.around(sigma,3))
legends.append("True value")
pylab.legend(legends)
pylab.grid(True)
pylab.show()

# ***********************************QUESTION 5***************************************

# print(t[::5].shape)
# print(data[:,1][::5].shape)
# print(sigma[0])

pylab.plot(t,g(t,true_A,true_B),color='black')
pylab.errorbar(t[::5],data[:,1][::5],sigma[0],fmt='.')
pylab.title('Error bars for Column 1 data with sigma = 0.1')
pylab.xlabel('t')
pylab.ylabel('f(t) + noise')
pylab.legend(['True Value','error_bar'])
pylab.grid(True)
pylab.show()

# ***********************************QUESTION 6***************************************

J = sp.jn(2, t)
M = pylab.c_[J,t]
P0 = np.array([[true_A],[true_B]])
# print(np.dot(M,P0)==np.reshape(g(t,true_A,true_B),(101,1)))
# print(np.dot(M,P0).shape)
# print(g(t,true_A,true_B).shape)

# np.allclose method is used to compare arrays woth floating point dtype to avoid rounding errors

print(np.allclose(np.dot(M,P0),np.reshape(g(t,true_A,true_B),(101,1))))

# ***********************************QUESTION 7***************************************

A = np.linspace(0,2,21)
B = np.linspace(-0.2,0,21)
errors = np.zeros((21,21))
for i in range(len(A)):
    for j in range(len(B)):
        errors[i,j] = mean_squared_error(data[:,1],g(t,A[i],B[j])) 

# ***********************************QUESTION 8***************************************

cp = pylab.contour(A,B,errors,20)
pylab.plot(1.05,-0.105,"ro")
pylab.annotate(r"$Exact\ location$",xy=(1.05,-0.105))
pylab.clabel(cp,inline=True)
pylab.xlabel(r"$A$",size=20)
pylab.ylabel(r"$B$",size=20)
pylab.title(r"Q8:Countour plot for $\epsilon_{ij}$")
pylab.show() 

# ***********************************QUESTION 9 & 10***************************************

A_error=[]      # Store error in value of A
B_error=[]      # Store error in value of B
Mean_errors=[]   # To store errors for each data column
for l in range(1,10):
    p,_,_,_ = lstsq(M,data[:,l])

    Mean_errors.append(mean_squared_error(np.dot(M,p),data[:,l]))
    A_error.append(abs(true_A- p[0]))
    B_error.append(abs(true_B - p[1]))


pylab.grid()
pylab.plot(sigma,A_error,'o--')
pylab.plot(sigma,B_error,'o--')
pylab.title('A_error and B_error for different sigma values(Linear)')
pylab.xlabel('Sigma') 
pylab.ylabel('A_error and B_error')
pylab.legend(['A_error','B_error'])
pylab.show()

# ***********************************QUESTION 11***************************************

pylab.grid()
pylab.loglog(sigma,A_error,'o')
pylab.loglog(sigma,B_error,'o')
pylab.errorbar(sigma,A_error,yerr=0.1,fmt = 'o')
pylab.errorbar(sigma,B_error,yerr=0.1,fmt = 'o')
pylab.title('A_error and B_error for different sigma values(loglog)')
pylab.xlabel('Sigma')
pylab.ylabel('A_error and B_error in logscale')
pylab.legend(['A_error in logscale ','B_error in logscale'])
pylab.show()


pylab.grid()
pylab.plot(sigma,Mean_errors,'o--')
pylab.title('MSError for f(t) different values of A and B(linear)')
pylab.xlabel('sigma')
pylab.ylabel('MSError')
pylab.show()

pylab.grid()
pylab.loglog(sigma,Mean_errors,'o--')
pylab.title('MSError of f(t) for different values of A and B(loglog)')
pylab.xlabel('sigma in logscale')
pylab.ylabel('MSError in logscale')
pylab.show()

    





