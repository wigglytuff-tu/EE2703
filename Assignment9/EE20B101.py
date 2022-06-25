'''
--------------------------------------------------------------------------------------------------
Assignment 5 - EE2703 (Jan-May 2022)
Purvam Jain (EE20B101)
--------------------------------------------------------------------------------------------------
'''

# Import Necessary Libraries

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import cm
from numpy.fft import fft, fftshift
from numpy.random import uniform, randn
import mpl_toolkits.mplot3d.axes3d as p3
import argparse as ag  

pi = np.pi  # Define constant

# Defining Mathematical functions used

func1 = lambda t: np.sin(np.sqrt(2)*t)
func2 = lambda t: np.sin(1.25*t)
cosq = lambda t: np.cos(0.86*t)**3
cosd = lambda t: np.cos(w0*t + delta)
cos_noisy = lambda t: np.cos(w0*t + delta) + 0.1*randn(N)
chirp = lambda t: np.cos(16*t*(1.5 + t/(2*pi)))


# DEfining transform Generator

def fft_generator(f, ii_cutoff = 1e-3, use_win = True, anti_sym = False):	
	x = num_cycles*np.linspace(-1*pi, pi, N+1)[:-1]         # Define input range
	y = f(x)
	F_sample = N/num_cycles          # Calculating sampling frequency
	if use_win == True:               # Check if applying windowing
		window = hamming(N)
		y = y*window    
	y = fftshift(y)                 # Apply fourirer transform
	if anti_sym == True:
		y[0] = 0
	Y = fftshift(fft(y))/N
	w = np.linspace(-0.5, 0.5, N+1)*F_sample; w = w[:-1]        # Frequency range
	ii = np.where(abs(Y)>ii_cutoff)
	return y, Y, w, ii

# Function to plot our graphs
def plotter(Y, w, ii, x_range, title = "DFT Magnitude and Phase"):
	fig = plt.figure()
	plt.subplot(2,1,1)
	plt.xlim(x_range)
	plt.plot(w, abs(Y), lw = 2)
	plt.ylabel(r"|Y|", size = 16)
	plt.grid(True)
	plt.title(title)
	plt.subplot(2,1,2)
	plt.xlim(x_range)
	plt.plot(w[ii], (np.angle(Y[ii])), 'ro', lw = 2)
	plt.ylabel(r"Phase of Y")
	plt.grid(True)
	plt.tight_layout()
	plt.show()
	return 0

# Function to implement Hamming Window 

def hamming(N, alpha = 0.54):
    N = int(N)
	# Class of hamming windows may be obtained by varying alpha (alpha = 0.5 is called Hann window)
	# Hamming window takes value 1 at index (N-1)/2, if the y- vector is rearranged using fftshift, value 1 
	# should correspond to index 0 (alpha + (1-alpha)*np.cos(2*pi*x/(N-1)))
    t = np.linspace(0, N-1, N)
    return fftshift(alpha + (1-alpha)*np.cos(2*pi*t/(N-1)))

# Function to estimate omega and delta

def estimator(f, title):
	y, Y, w, ii = fft_generator(f)
	est_idx = np.argmax(abs(Y[64:]))
	w_est = abs(w[est_idx+64])	
	delta_est = np.angle(Y[est_idx+64])
	print("\nOmega estimate = {0} \nActual Omega = {1} (Error = {2})".format(w_est, w0, round(abs(w_est - w0), 4)))
	print("Delta Estimate = {0} \nActual Delta =  {1} (Error = {2})".format(delta_est, delta, round(abs(delta_est - delta), 4)))
	plotter(Y, w, ii, [-10, 10], title)
	return 0

# Function to generate 3D surface plots

def surfer(y, x, z, xlabel, ylabel):
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    Y, X = np.meshgrid(y, x)
    surf = ax.plot_surface(X, Y, z, cmap = cm.jet)
    fig.colorbar(surf, shrink=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig

# Argument Parsing Block
parser = ag.ArgumentParser(formatter_class = ag.RawTextHelpFormatter)
parser.add_argument("--fn_choice", type=int, default=1, help="Choose time domain function:\n1: Sinusoids\
	\n2: cos^3(0.86t)\n3: Estimation of frequency and phase of a sinusoid\n4: Chirped Signal- DFT\
	\n5: Chirped Signal- Time-Frequency Plot\n6: Accuracy analysis of w0 and delta estimation")
parser.add_argument("-N", type=int, default=64, help="Enter number of samples.")
parser.add_argument("-cycles", type=int, default=4, help="Enter number of cycles.")
params = parser.parse_args()

fn_choice = params.fn_choice
N = params.N 
num_cycles = params.cycles 

if fn_choice == 1:
	y, Y, w, ii = fft_generator(func1, anti_sym = True)
	plotter(Y, w, ii, [-10, 10], r'DFT Mag and Phase of $sin(\sqrt{2}t)$')
	y, Y, w, ii = fft_generator(func2, anti_sym = True)
	plotter(Y, w, ii, [-10, 10], r'DFT Mag and Phase of $sin(\frac{5}{4}t)$')
elif fn_choice == 2:
	#num_cycles = 8
	y, Y, w, ii = fft_generator(cosq, use_win = True)
	plotter(Y, w, ii, [-10, 10], r'DFT Mag and Phase of $cos^3(\omega_ot)$, with Hamming window')
	y, Y, w, ii = fft_generator(cosq, use_win = False)
	plotter(Y, w, ii, [-10, 10], r'DFT Mag and Phase of $cos^3(\omega_ot)$, without Hamming window')
elif fn_choice == 3:
	#N = 128; num_cycles = 32
	w0 = round(uniform(0.5, 1.5), 3)	# Generate w0 and delta by sampling from a uniformly random distribution
	delta = round(uniform(-pi, pi), 3)
	print("Without Noise:")
	estimator(cosd, r'DFT Mag and Phase of $cos(\omega_ot + \delta)$, without Gaussian Noise')
	print("With Noise:")
	estimator(cos_noisy, r'DFT Mag and Phase of $cos(\omega_ot + \delta)$, with Gaussian Noise')
elif fn_choice == 4:
	N = 1024; num_cycles = 1
	y, Y, w, ii = fft_generator(chirp)
	plotter(Y, w, ii, [-64, 64], r'DFT Mag and Phase of $cos(16t(1.5 + \frac{t}{2\pi})$')
elif fn_choice == 5:
	F_sample = N/num_cycles
	x = num_cycles*np.linspace(-1*pi, pi, N+1)[:-1]
	y = chirp(x)
	y_2d = np.reshape(y, (16, 64)).T
	Y_dft = np.zeros(y_2d.shape, dtype = complex)
	N_2d = N/16
	w = np.linspace(-0.5, 0.5, int(N_2d+1))*F_sample; w = w[:-1]
	for i in range(16):
		y_2d[:, i] *= hamming(N_2d)
		Y_dft[:, i] = fftshift(fft(fftshift(y_2d[:, i])))/N_2d
	fig1 = surfer(np.linspace(0, 15, 16), w, abs(Y_dft), "Freq Axis", "Time Bins")
	fig2 = surfer(np.linspace(0, 15, 16), np.linspace(0, 63, 64), y_2d, "Time Axis", "Time Bins")
	plt.show()
elif fn_choice == 6:
	print("Accuracy of w0 and delta estimation:")
	werr = []; del_err = []
	del_mse = 0; flag = 0
	#N = 128; num_cycles = 32
	for i in range(10000):
		w0 = round(uniform(0.5, 1.5), 3)
		delta = round(uniform(-pi, pi), 3)
		y, Y, w, ii = fft_generator(cos_noisy)
		est_idx = np.argmax(abs(Y[64:]))
		w_est = abs(w[est_idx+64])	# argmax finds returns the first index if two or more indices have the max value, which corresponds to -ve w
		delta_est = np.angle(Y[est_idx+64])
		werr.append(abs(w_est- w0))
		if (abs(delta_est - delta)>6):
			flag += 1
		del_err.append(abs(delta_est - delta))
		del_mse += (delta_est - delta)**2
	print("Max w error = {0}\nMax delta error = {1}, Number of 2pi occurrences = {2}\
		\nDelta MSE = {3}".format(np.max(werr), np.max(del_err), flag,del_mse/10000))