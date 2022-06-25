'''
--------------------------------------------------------------------------------------------------
EndSem - EE2703 (Jan-May 2022)
Purvam Jain (EE20B101)
--------------------------------------------------------------------------------------------------
'''
# Import necessary Libraries

import numpy as np
import argparse as ag
import matplotlib.pyplot as plt

# Argument Parser block

parser = ag.ArgumentParser(formatter_class = ag.RawTextHelpFormatter)
parser.add_argument("-N", type=int, default=100, help="Enter Number of Sections in each half section of the antenna.")
parser.add_argument("-r", type=int, default=0.01, help="Enter radius of wire.")
parser.add_argument("-l", type=int, default=0.5, help="Quarter Wavelength")
params = parser.parse_args()


# Defining given parameters as global variables
pi = np.pi
l = params.l     # Quarter Wavelength
c = 2.9979e8    # Speed of Ligth
mu0 = (4e-7)*pi     # Magnetic permeabiltiy constant
N = params.N         # Number of Sections in each half section of the antenna
Im = 1.0        # current injected into the antenna
a = params.r        # radius of wire
Lambda = 4*l        # wavelength
f = c/Lambda        # Frequency
k = 2*pi/Lambda     # Wave Number
dz = l/N            # spacing of current samples
    
# ***********************************QUESTION 1***************************************
'''
Pseudo Code #1:
1. Generate Array z = i*dz where -N<=i<=N ; points where we compute currents
2. Use arange funtion to generate an array of 2N + 1 elements from -N to N 
3. Generate array u of 2N - 2 length of locations of unknown currents
4. For this copy array z and remove known elements 0: at ends and Im at centre
5. Construct the current vector I at points corresponding to
vector z, and the current vector J at points corresponding to vector u.
6. Reshape the array into column vectors.
7. As instructed we use a if condition to print outputs only if N<10.
'''
z = np.arange(-N,N+1)*dz        # Generate array with integer values from -N to N
u = z.copy()                    # Copy z array into u
u = np.delete(u,[0,N,-1])       # Delete boundary points and central current
I = np.zeros(2*N+1)             # ZEro array with 2N+1 indices
I[0],I[-1],I[N] = 0,0,Im        # Specify known current values
J = np.zeros(2*N-2)             # Empty array of 2N-2 zeroes

z = z.reshape((len(z),1))       # Turn defined arrays into column vectors
u = u.reshape((len(u),1))
I = I.reshape((len(I),1))
J = J.reshape((len(J),1))

if N<10:                        # Print only for small N values                
    print("z:\n",z)
    print("u:\n",u)
    print("I:\n",I)
    print("J:\n",J)



# ***********************************QUESTION 2***************************************

'''
Pseudo Code #2:
1. Define a function to Generate M matrix
2. We take N and r(radius) as inputs to this function
3. Use identity function in numpy library to make indentity matrix
4. Print the matrix rounded to two decimal places if N,10
5. Return M to user
6. Implement the function on default values
'''
def compute_M(N,r):
    M = np.identity(2*N-2)/(2*pi*r)         # Use identity function to generate indentity matrix
    if N<10:
        print("M:\n",M.round(2))
    return M
M = compute_M(N,a)

# ***********************************QUESTION 3***************************************
'''
Pseudo Code #3:
1. Define function to generate Ru and Rz matrices which takes inputs as radius, z and u vectors
2. Define meshgrids on z and u vectors
3. To compute distances Rij we take maginutde of rsultant vector of R = r + z-z'
4. Print rounded matrices if N<10
5. Return results upon calling function 
'''
def compute_Ruz(r,z,u):
    zi,zj = np.meshgrid(z,z)            
    ui,uj = np.meshgrid(u,u)
    Rz = np.sqrt(r**2 + (zj-zi)**2)         # Calculating Distance/Magnitude of resultant vector
    Ru = np.sqrt(r**2 + (uj-ui)**2)
    if N<10:
        print("Ru:\n",Ru.round(2))
        print("Rz:\n",Rz.round(2))
    return Ru,Rz

Ru,Rz = compute_Ruz(a,z,u)

'''
Pseudo Code #4:
1. Define function to compute P and Pb matrices which take RiN vector and Ru matrix as input
2. Implement formula to get Pb and P matrices
3. Print the rounded matrices if N<10 and return on implementation of function
4. Define Rin vector as Nth column of Rz matrix without the known values of 0,2N and N indices
5. Reshape the resultant array to get column vector
6. Implement the earlier defined function to get P and Pb
'''

def compute_Ps(Rin,Ru):
    Pb = (mu0/(4*pi))*(np.exp(-1j*k*Rin)/Rin)*dz
    Pb = Pb.reshape((len(Pb),1))
    P = (mu0/(4*pi))*(np.exp(-1j*k*Ru)/Ru)*dz
    if N<10:
        print("Pb:\n",(Pb*1e8).round(2))
        print("P:\n",(P*1e8).round(2))
    return Pb,P

Rin = Rz[:,N]                       
Rin = np.delete(Rin,[0,N,2*N])
# print(Rin)
Rin = Rin.reshape((len(Rin),1))
Pb,P = compute_Ps(Rin,Ru)

# ***********************************QUESTION 4***************************************
'''
Pseudo Code #5:
1. Define function to compute Qij and Qb matrices with inputs P,Pb,Rin,Ru and r as defined earlier
2. Implement formula as given to get Qb and Q matrices
3. Print of N<10 and return the values to user
'''
def compute_Q(P,Pb,r,Rin,Ru):
    Qb = Pb*(r/mu0)*((1j*k/Rin)+(1/Rin**2))
    Qb = Qb.reshape((len(Qb),1))
    Qij = P*(r/mu0)*((1j*k/Ru)+(1/Ru**2))
    if N<10:
        print("Qb:\n",(Qb).round(5))        # Rounded to 5 for accurate display
        print("Qij:\n",(Qij))
    return Qij,Qb
Qij,Qb = compute_Q(P,Pb,a,Rin,Ru)

# ***********************************QUESTION 5***************************************
'''
Pseudo Code #6:
1. Calculate J vector using inverse and matrix multiplication utilities of Numpy
2. Insert the missing values of boundary conditions and current at centre
3. Print rounded matrix if N<10
4. Concatenate the two-part original function into one original function
5. Plot the original and estimated functions
'''

J = np.matmul(np.linalg.inv(M-Qij),Qb)

J = np.insert(J,0,0)
J = np.insert(J,N,Im)
J = np.append(J,0)

if N<10:
    print("J:\n",J)

f1 = Im*np.sin(k*(l-z[N:]))     # For positive half
f2 = Im*np.sin(k*(l+z[:N]))     # For negative half
fx = np.append(f2,f1)

plt.grid()
plt.plot(z,J)
plt.plot(z,fx)
plt.title('Assumed Function and Estimated Function')
plt.xlabel('z')
plt.ylabel('I')
plt.legend(["Estimated","Assumed"])
plt.show()   


  

