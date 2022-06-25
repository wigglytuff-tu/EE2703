'''
--------------------------------------------------------------------------------------------------
Assignment 5 - EE2703 (Jan-May 2022)
Purvam Jain (EE20B101)
--------------------------------------------------------------------------------------------------
'''

# import necessary libraries 

import argparse
import numpy as np 
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftshift, ifftshift

pi = np.pi  # Define pi constant for future use

# Lambda function definitions:

sin = lambda t: np.sin(5*t) 
amp_mod = lambda t: (1+0.1*np.cos(t))*np.cos(10*t)
sin_cube = lambda t: np.sin(t)**3
cos_cube = lambda t: np.cos(t)**3
freq_mod = lambda t: np.cos(20*t + 5*np.cos(t))
gaussian = lambda t: np.exp(-(t**2)/2)
gauss_ft = lambda w: np.exp(-((w**2)/2))*(np.sqrt(2*pi))	 # Fourier transform of gaussian

# Define Transform Generator

def fft_generator(f, num_cycles, sample_freq, ii_cutoff):	
	N = num_cycles*sample_freq							# Number of samples
	x = np.linspace(-1*pi, pi, N+1)*num_cycles; x = x[:-1]		# Range
	y = f(x)
	y = ifftshift(y)				# Inverse fourier transform
	Y = fftshift(fft(y))/N			# Fourier transform
	w = np.linspace(-0.5, 0.5, N+1)*sample_freq; w = w[:-1]		# Frequency range
	ii = np.where(abs(Y)>ii_cutoff)
	return Y, w, ii

# Define function for plotting generated transforms

def plotter(Y, w, ii, x_range, title, plot_gauss = False):
	fig = plt.figure()
	plt.subplot(2,1,1)
	plt.xlim(x_range)
	plt.plot(w, abs(Y), lw = 2)
	if plot_gauss == True:
		plt.plot(w, gauss_ft(w))
		plt.legend(["Estimated FFT", "True FFT"])
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
	# Visualizing error in gaussian transform
	if plot_gauss == True:
		plt.xlim(x_range)
		plt.plot(w, np.real(Y - gauss_ft(w)))
		plt.title("Error of DFT with true Fourier Transform")
		plt.show()
		print(max(abs(np.real(Y - gauss_ft(w)))))			# Computing max error is estimated transform and actual function
	return 0

# Argument Parsing Block
parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)
parser.add_argument("-fn_choice", type=int, default=5, help="Choose time domain function:\n1: sin(5t)\
	\n2: (1+0.1cos(t))cos(10t) (AMPLITUDE MODULATION)\n3: sin^3(t), cos^3(t)\n4: cos(20t + 5cos(t)) (FREQUENCY MODULATION)\
	\n5: exp(-(t^2)/2) (GAUSSIAN)")
parser.add_argument("-F_sample", type=int, default=64, help="Enter sampling frequency.")
parser.add_argument("-cycles", type=int, default=4, help="Enter number of cycles.")
params = parser.parse_args()
fn_choice = params.fn_choice
sampling_freq = params.F_sample
num_cycles = params.cycles
if fn_choice == 1:
	Y, w, ii = fft_generator(sin, num_cycles, sampling_freq, 1e-3)
	plotter(Y, w, ii, [-15, 15], "Spectrum of sin(5t)")
elif fn_choice == 2:
	Y, w, ii = fft_generator(amp_mod, num_cycles, sampling_freq, 1e-3)
	plotter(Y, w, ii, [-15, 15], "Spectrum of (1+0.1cos(t))cos(10t) (AM signal)")
elif fn_choice == 3:
	Y1, w1, ii1 = fft_generator(sin_cube, num_cycles, sampling_freq, 1e-3)
	plotter(Y1, w1, ii1, [-15, 15], "Spectrum of sin^3(t)")
	Y2, w2, ii2 = fft_generator(cos_cube, num_cycles, sampling_freq, 1e-3)
	plotter(Y2, w2, ii2, [-15, 15], "Spectrum of cos^3(t)")
elif fn_choice == 4:
	Y, w, ii = fft_generator(freq_mod, num_cycles, sampling_freq, 1e-3)
	plotter(Y, w, ii, [-40, 40], "Spectrum of cos(20t + 5cos(t)) (FM signal)")
elif fn_choice == 5:
	sampling_freq = params.F_sample
	num_cycles = params.cycles
	Y, w, ii = fft_generator(gaussian, num_cycles, sampling_freq, 1e-6)
	Y = Y*num_cycles*2*pi
	plotter(Y, w, ii, [-6*pi, 6*pi], "Spectrum of Gaussian", True)
