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
theta_h = 0.0 * np.pi

# Gilbert damping and spin pump damping
alpha0 = 0.05
alphaprime = 0.01
alpha = alpha0 + alphaprime

# Microwave field (omega is end of range for-loop)
hmicro = 0.1

# Defining H field
hend = 600
Ms0 = 1.6 # Saturation demagnitization 
Dx, Dy, Dz = [0 , -0.01, -0.99] # Demagnization tensor
begin_h = 0
step_h = 1

# Frequency microwave field
f = 120

# Make empty arrays for sweep of frequencies in for-loop for code efficiency
maxx = np.zeros(int((hend - begin_h) / step_h))
maxy = np.zeros(int((hend - begin_h) / step_h))
maxz = np.zeros(int((hend - begin_h) / step_h))

fields = range(begin_h, hend, step_h)

for h in fields:
    def dUdt(U, t):
        # Absorbing mx, my, mz into vector U
        mx, my, mz = U
    
        # Defining Hext vector
        Hextx, Hexty, Hextz = [(h/(2000)) * np.cos(theta_h) , (h/(2000)) * np.sin(theta_h) , 0]
        # Defining Heff vector
        # Addition of constant term, microwave term and demagnization tensor term
        Hx = Hextx + Dx * Ms0 * mx 
        Hy = Hexty + Dy * Ms0 * my 
        Hz = Hextz + Dz * Ms0 * mz + hmicro * np.cos(f*t)



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
    A_r = 10**15 
    dmdt = np.diff(magnetization)
    I_spin = sp.constants.hbar / (4 * np.pi) * A_r * np.cross(magnetization, dmdt)

    """
    Eq. (1) in paper [1]
    """
    # Choose spin polarization vector sigma along constant field # H_ext
    sigma = [np.cos(theta_h) , np.sin(theta_h) , 0]

    # Inverse spin hall efficiency, chosen arbitrarily for now
    D_ISHE = 0.5
    I_charge = D_ISHE * np.cross(I_spin, sigma)

    # Calculate amplitude fmr precession and put this in empty array
    ampx = max(I_charge[round(hend*0.65) : (hend-1) , 0])
    ampy = max(I_charge[round(hend*0.65) : (hend-1) , 1])
    ampz = max(I_charge[round(hend*0.65) : (hend-1) , 2])
    
    # Store iteration in x,y,z variable arrays
    maxx[h] = ampx
    maxy[h] = ampy
    maxz[h] = ampz

xas = []
for x in fields:
    xas.append(x/2)


"""
Plot Max Amplitude against the frequency
""" 
fig = plt.figure(figsize = plt.figaspect(.4))
fig.suptitle(r'Left: maximum amplitude charge current. Right: derivative $dI(H)/dH$')
ax = fig.add_subplot(1 , 2 , 1)

# Grid 
ax.grid(True)

# Plotting frequencies
ax.plot(xas, maxx, label = r'$I_x$')
ax.plot(xas, maxy, label = r'$I_y$')
ax.plot(xas, maxz, label = r'$I_z$')

# Legend
ax.legend()


# Labels
ax.set_xlabel(r'Field $H$ [mT]')
ax.set_ylabel(r'Maximum amplitude [a.u.]')
ax.tick_params(labelleft=False)    


"""
Plot DI(H)/DH
"""
DIDHx = np.diff(maxx)
DIDHy = np.diff(maxy)
DIDHz = np.diff(maxz)

ax = fig.add_subplot(1, 2, 2)

ax.grid(True)


# Removing 1 frequency element because np.diff() wil be one element shorter
# If this is not done, x and y(x) dimensions will not match
xas_cut = xas[:-1]

# X axis range
plt.xlim(10, 300)
plt.ylim(-1.5*10**(-23),1*10**(-23))

# Plotting
#ax.plot(xas_cut, DIDHx, label = r'$I_x$')
ax.plot(xas_cut, DIDHy, label = r'$I_y$')
#ax.plot(xas_cut, DIDHz, label = r'$I_z$')

# Legend
ax.legend()

# Labels
ax.set_xlabel(r'Field H [mT]')
ax.set_ylabel(r'Charge $dI(H)/dH$ [a.u.]')
ax.tick_params(labelleft=False)    

# Saving
plt.savefig('figuren/amplitudeplot_and_DIDH_aprime0_01.pdf' , bbox_inches = 'tight')