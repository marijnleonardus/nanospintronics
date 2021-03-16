# -*- coding: utf-8 -*-
"""
Created on Mon May 21 10:30 2020

@author: Roy Rosenkamp, Marijn Venderbosch
"""

# Importing packages
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
# Last import registers the 3D projection, but is otherwise unused 

# Constants in units of [10^6 A/m] from mathematica notebook '3MN220 LLG model.nb'
gamma = 176
mu0 = 0.4 * np.pi
theta_m = 0.2

# Gilbert damping and spin pump damping
alpha0 = 0.05
alphaprime = 0.01
alpha = alpha0 + alphaprime

# Microwave field (omega is end of range for-loop)
h = 0.1
omega = 400

# Defining H field

Ms0 = 1.6 # Saturation demagnitization 
Hextx, Hexty, Hextz = [0.2 , 0 , 0] # Constant external field 
Dx, Dy, Dz = [0 , -0.01, -0.99] # Demagnization tensor

# Frequency speeping range
begin_f = 0
step_f = 1

# Make empty arrays for sweep of frequencies in for-loop for code efficiency
maxx = np.zeros(int((omega - begin_f) / step_f))
maxy = np.zeros(int((omega - begin_f) / step_f))
maxz = np.zeros(int((omega - begin_f) / step_f))

frequencies = range(begin_f, omega, step_f)

for f in frequencies:
    def dUdt(U, t):
        # Absorbing mx, my, mz into vector U
        mx, my, mz = U
    
        # Defining Heff vector
        # Addition of constant term, microwave term and demagnization tensor term
        Hx = Hextx + Dx * Ms0 * mx 
        Hy = Hexty + Dy * Ms0 * my 
        Hz = Hextz + Dz * Ms0 * mz + h * np.cos(f*t)



        # For RHS see mathematica file
        dmxdt = -gamma * mu0 / (1 + alpha**2) * ( Hz * my - Hy * mz + alpha * ( Hy * mx * my - Hx * my**2 + Hz * mx * mz - Hx * mz**2))
        dmydt = -gamma * mu0 / (1 + alpha**2) * (-Hz * mx + Hx * mz + alpha * (-Hy * mx**2 + Hx * mx * my + Hz * my * mz - Hy * mz**2))
        dmzdt = -gamma * mu0 / (1 + alpha**2) * ( Hy * mx - Hx * my + alpha * (-Hz * mx**2 - Hz * my**2 + Hx * mx * mz + Hy * my * mz))

        return [dmxdt, dmydt, dmzdt]

    # Create time vector. Start, stop and steps:
    t_span = np.linspace(0 , 2 , 1001)

    # Initial conditions. Alignment of vector m(t=0) under angle theta in x-y-plane
    mx0 = np.cos(theta_m) 
    my0 = 0
    mz0 = np.sin(theta_m) 
    Uzero = [mx0 , my0 , mz0]

    # Solve system of equations DUdt for initial conditon Uzero and time range t_span
    magnetization = odeint(dUdt, Uzero, t_span)

    """
    Eq. (1) in paper [2]
    """
    # Spin current conductance, chosen arbitrarily for now
    A_r = 10**10 
    dmdt = np.diff(magnetization)
    I_spin = sp.constants.hbar / (4 * np.pi) * A_r * np.cross(magnetization, dmdt)

    """
    Eq. (1) in paper [1]
    """
    # Choose spin polarization vector sigma along constant field # H_ext
    sigma = [1 , 0 , 0]

    # Inverse spin hall efficiency, chosen arbitrarily for now
    D_ISHE = 0.5
    I_charge = D_ISHE * np.cross(I_spin, sigma)

    # Calculate amplitude fmr precession and put this in empty array
    ampx = max(I_charge[100 : (omega-1) , 0])
    ampy = max(I_charge[100 : (omega-1) , 1])
    ampz = max(I_charge[100 : (omega-1) , 2])
    
    # Store iteration in x,y,z variable arrays
    maxx[f] = ampx
    maxy[f] = ampy
    maxz[f] = ampz

"""
Plot Max Amplitude against the frequency
""" 
fig = plt.figure(figsize = plt.figaspect(.4))
fig.suptitle(r'Amplitude Charge Current')
ax = fig.add_subplot(1 , 1 , 1)

# Grid and legend
ax.grid(True)
ax.legend()

# Plotting frequencies
ax.plot(frequencies, maxx, label = r'$I_x$')
ax.plot(frequencies, maxy, label = r'$I_y$')
ax.plot(frequencies, maxz, label = r'$I_z$')

# Labels
ax.set_xlabel(r'Frequency')
ax.set_ylabel(r'Amplitude')

# Saving
plt.savefig('figuren/amplitudeplot.pdf' , bbox_inches = 'tight')
