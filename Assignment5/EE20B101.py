'''
--------------------------------------------------------------------------------------------------
Assignment 5 - EE2703 (Jan-May 2022)
Purvam Jain (EE20B101)
--------------------------------------------------------------------------------------------------
'''


import argparse
import numpy as np
from pylab import c_,cm
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

parser = argparse.ArgumentParser(description="Optional inputs for Nx,Ny,Niter,radius. Allowed values: radius*2<Nx and Nx=Ny")

parser.add_argument('-Nx', type=int, default=25, help="size along x")
parser.add_argument('-Ny', type=int, default=25, help="size along y")
parser.add_argument('-radius', type=int, default=8, help="radius of central lead")
parser.add_argument('-Niter', type=int, default=1500, help="number of iterations to perform")
args = parser.parse_args()

Nx = args.Nx  # size along x
Ny = args.Ny # size along y
radius = args.radius # radius of central lead
Niter = args.Niter # number of iterations to perform

phi = np.zeros((Ny,Nx),dtype=float)
x = np.linspace(-0.5, 0.5, Nx)	# since given plate is 1cmX1cm and middle region should be set to x=0,y=0
y = np.linspace(-0.5, 0.5, Ny)
Y,X = np.meshgrid(y,x)
scaled_radius = (radius/2)/(Nx//2) + 0.0025*radius
''' scaled_radius: Since given plate is of 1cmX1cm and centre is taken as origin Nx//2 = 0.5cm
hence input radius in that scale will be (radius/2)/(Nx//2) to this we add a tolerance value of 
0.25% of radius which is derived from default values given to us.'''
ii = np.where(X*X + Y*Y <= scaled_radius*scaled_radius)
phi[ii] = 1.00 # all i,j values inside circle made equal to 1
xn, yn = ii

'''Similar to scaled radius we have to scale and shift the points xn and yn too.'''
xn = (((xn-Nx//2)/2)/(Nx//2))
yn = (((yn-Ny//2)/2)/(Ny//2))

# Plot a contour plot of the potential in Figure 1. Mark the V = 1 region by marking those nodes red.
plt.grid()
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(xn,yn,'ro')
plt.contourf(X,Y,phi)
plt.legend(['V=1'])
plt.title('Original Potential')
plt.colorbar()
plt.show()

errors = np.zeros(Niter)
for k in range(Niter):
	oldphi = phi.copy()   # phi copied to new memory area
	phi[1:-1, 1:-1] = 0.25*(phi[1:-1, 0:-2] + phi[1:-1, 2:] + phi[0:-2, 1:-1] + phi[2:, 1:-1])	#Poisson Update
	phi[0,1:-1],phi[-1,1:-1],phi[:,-1] = phi[1,1:-1],phi[-2,1:-1],phi[:,-2]
    # Boundary Condition for Left Surface, Right Surface and Top Surface respectively
	phi[ii] = 1.0 # Circular boundary
	errors[k] = (abs(phi - oldphi)).max()

'''how the errors are evolving in a semilog and loglog plot.'''

plt.grid()
plt.title("Error on a semilog plot")
plt.xlabel("No of iterations")
plt.ylabel("Error")
plt.semilogy(range(Niter),errors)
plt.show()

plt.grid()
plt.title("Error on a loglog plot")
plt.xlabel("No of iterations")
plt.ylabel("Error")
plt.loglog((np.asarray(range(Niter))+1),errors)
plt.loglog((np.asarray(range(Niter))+1)[::50],errors[::50],'ro')
plt.legend(["real","every 50th value"])
plt.show()


'''Extract the fit for the entire vector of errors and for those error entries after the 500th iteration.
Plot the fit in both cases on the error plot itself.'''
# we use linear squares to find best fit curve
c_approx = lstsq(c_[np.ones(Niter),np.arange(Niter)],np.log(errors))
a, b = c_approx[0][0], c_approx[0][1]
print("The values of A and B are: ",np.exp(a),b)

c_approx_500 = lstsq(c_[np.ones(Niter-500),np.arange(500,Niter)],np.log(errors[500:]))
a_500,b_500 = c_approx_500[0][0],c_approx_500[0][1]
print("The values of A and B for the iterations after 500 are: ",np.exp(a_500),b_500)



plt.grid()
plt.title("Best fit for error on a loglog scale")
plt.xlabel("No of iterations")
plt.ylabel("Error")
ex = np.asarray(range(Niter))+1
plt.loglog(ex,errors)
plt.loglog(ex[::50],np.exp(a+b*np.asarray(range(Niter)))[::50],'ro')
plt.loglog(ex,np.exp(a_500+b_500*np.asarray(range(Niter))),'g')
plt.legend(["errors","fit1","fit2"])
plt.show()

# A 3-D plot of the potential.
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, phi,rstride=1, cstride=1,cmap = cm.jet,linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('3-D Surface plot of the potential')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# A contour plot of the potential.
plt.grid()
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(xn,yn,'ro')
plt.contourf(X,Y,phi)
plt.legend(['V=1'])
plt.title('Contour plot of potential')
plt.colorbar()
plt.show()

# Obtain the currents. Plot the current density using quiver, and mark the electrode via red dot.

Jx,Jy = np.zeros_like(phi),np.zeros_like(phi)

plt.grid()
Jy[:,1:-1] = 0.5*(phi[:,:-2] - phi[:,2:])
Jx[1:-1] = 0.5*(phi[:-2] - phi[2:])
plt.scatter(xn,yn,s=4,color='r',marker='o')

plt.quiver(X,Y,-Jx[::-1,:],Jy[::-1,:],scale=5)
plt.title('Quiver plot of the current densities')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['Electrode','Current density'])
plt.show()

#  Take the currents calculated above and plot the heat map

J_sq = Jx**2 + Jy**2 # Energy Dissipated
plt.grid()
plt.title('Contour plot of the heat generated')
cp = plt.contourf(X,Y,J_sq,10)
plt.clabel(cp,fontsize = 8,colors='r')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.show()


