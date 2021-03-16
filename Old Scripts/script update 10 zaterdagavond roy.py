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
from progressbar import ProgressBar
pbar = ProgressBar()
# Last import registers the 3D projection, but is otherwise unused 

# Constants in units of [10^6 A/m] from mathematica notebook '3MN220 LLG model.nb'
gamma = 176
mu0 = 0.4 * np.pi
theta_m = 0.2
theta_h = 0.5 * np.pi

# Gilbert damping and spin pump damping
alpha0 = 0.05
alphaprime = 0.01
alpha = alpha0 + alphaprime

# Microwave field
hmicro = 0.1
f = 120

# Defining H field
end_h = 300
begin_h = 0
steps = 2 # number of steps == end_h*steps
Ms0 = 1.6 # Saturation demagnitization 
Dx, Dy, Dz = [0 , -0.01, -0.99] # Demagnization tensor

# Make empty arrays for sweep of frequencies in for-loop for code efficiency and define end for-loop
h_end_step = end_h * steps # defines end of for-loop (later, the field is divided by steps again)
maxx = np.zeros(int(h_end_step - begin_h))
maxy = np.zeros(int(h_end_step - begin_h))
maxz = np.zeros(int(h_end_step - begin_h))

# Empty array for DC field component
dccurrentx = np.zeros(int(h_end_step - begin_h))
dccurrenty = np.zeros(int(h_end_step - begin_h))
dccurrentz = np.zeros(int(h_end_step - begin_h))

fields = range(begin_h, h_end_step)

for h in pbar(fields):
    def dUdt(U, t):
        # Absorbing mx, my, mz into vector U
        mx, my, mz = U
    
        # Defining Hext vector, h/(1000*steps) --> (mT->T) and makes sure that for loop goes until end_h
        Hextx, Hexty, Hextz = [(h/(1000*steps)) * np.cos(theta_h) , (h/(1000*steps)) * np.sin(theta_h) , 0]
        # Defining Heff vector
        # Addition of constant term, microwave term and demagnization tensor term
        Hx = Hextx + Dx * Ms0 * mx 
        Hy = Hexty + Dy * Ms0 * my 
        Hz = Hextz + Dz * Ms0 * mz + hmicro * np.cos(f * t)

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

    # Calculate amplitude fmr precession 
    maxx[h] = max(I_charge[round(h_end_step*0.65) : (h_end_step-1) , 0])
    maxy[h] = max(I_charge[round(h_end_step*0.65) : (h_end_step-1) , 1])
    maxz[h] = max(I_charge[round(h_end_step*0.65) : (h_end_step-1) , 2])
    
    # Calculate DC charge current, which is simply the average of the charge current
    dccurrentx[h] = np.average(I_charge[round(h_end_step*0.65) : (h_end_step-1),0])
    dccurrenty[h] = np.average(I_charge[round(h_end_step*0.65) : (h_end_step-1),1])
    dccurrentz[h] = np.average(I_charge[round(h_end_step*0.65) : (h_end_step-1),2])
   
# Make array for x-axis plots, from 0 to hend instead of hend*steps, which is the range of the for-loop
xas = []
for x in fields:
    xas.append(x/steps)
   
"""
Plot Max Amplitude against the frequency
""" 
fig = plt.figure(figsize = plt.figaspect(.4))
fig.suptitle(r'Left: maximum amplitude AC charge current. Middle: derivative $dI^{AC}(H)/dH$. Right: DC Charge Current.')
ax = fig.add_subplot(1 , 3 , 1)

# Grid 
ax.grid(True)

# X- and Y- axis range
plt.xlim(50,300)
plt.ylim(-1.5*10**(-21),1.5*10**(-21))

# Plotting Max Amplitudes Charge current
ax.plot(xas, maxx, label = r'$I_x^{AC}$')
ax.plot(xas, maxy, label = r'$I_y^{AC}$')
#ax.plot(xas, maxz, label = r'$I_z^{AC}$')

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

ax = fig.add_subplot(1, 3, 2)

# Grid
ax.grid(True)

# Removing 1 frequency element because np.diff() wil be one element shorter
# If this is not done, x and y(x) dimensions will not match
xas_cut = xas[:-1]

# X- and Y- axis range
plt.xlim(50, 300)
plt.ylim(-1.5*10**(-23),1*10**(-23))

# Plotting
ax.plot(xas_cut, DIDHx, label = r'$I_x^{AC}$')
ax.plot(xas_cut, DIDHy, label = r'$I_y^{AC}$')
#ax.plot(xas_cut, DIDHz, label = r'$I_z{AC}$')

# Legend
ax.legend()

# Labels
ax.set_xlabel(r'Field H [mT]')
ax.set_ylabel(r'Charge $dI(H)/dH$ [a.u.]')
ax.tick_params(labelleft=False) 

"""
Plot DC Current
"""   
ax = fig.add_subplot(1 , 3 , 3)

# Grid 
ax.grid(True)

# X- and Y- axis range
plt.xlim(50, 300)
plt.ylim(0*10**(-21),5*10**(-21))

# Plotting DC Current
ax.plot(xas, dccurrentx, label = r'$I_x^{DC}$')
ax.plot(xas, dccurrenty, label = r'$I_y^{DC}$')
#ax.plot(xas, dccurrentz, label = r'$I_z^{DC}$')

# Legend
ax.legend()

# Labels
ax.set_xlabel(r'Field $H$ [mT]')
ax.set_ylabel(r'DC Current [a.u.]')
ax.tick_params(labelleft=False)    

# Saving
#plt.savefig('amplitudeplot_and_DIDH_and_DC-current.pdf' , bbox_inches = 'tight')