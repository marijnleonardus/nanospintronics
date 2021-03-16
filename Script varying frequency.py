# -*- coding: utf-8 -*-
"""
Created on Mon May 21 10:30 2020

THIS SCRIPT IS USED TO MAKE FIGURES FOR CHANGING MICROWAVE FREQUENCY

@author: Roy Rosenkamp, Marijn Venderbosch
"""

# Importing packages
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import odeint
#from mpl_toolkits.mplot3d import Axes3D
# Last import registers the 3D projection, but is otherwise unused
from scipy.optimize import curve_fit 
from progressbar import ProgressBar
pbar = ProgressBar()

# Constants in units of [10^6 A/m] from mathematica notebook '3MN220 LLG model.nb'
gamma = 176             # gyromagnetic ratio
mu0 = 0.4 * np.pi
theta_m = 0.2           # defines initial position magnetization vector
theta_h = 0.0 * np.pi   # angle external magnetic field

# Gilbert damping and spin pump damping
alpha0 = 0.05
alphaprime = 0.05
alpha = alpha0 + alphaprime

# Microwave field
hmicro = 0.1            # amplitude
f_start = 70            # start for-loop frequency
f_end = 250             # end for-loop frequency
f_startend = int((f_end-f_start)/10)    # defines number that is used in making the matrices in the end

# Defining H field
end_h = 600             # start for-loop for external field
begin_h = 0             # end for-loop
StepsPerField = 1       # == end_h*steps, choose integer
h_end_step = end_h * StepsPerField # defines end of for-loop (later, the field is divided by steps again (this to reduce step size))
Ms0 = 1.6               # Saturation demagnitization 
Dx, Dy, Dz = [0 , -0.01, -0.99] # Demagnization tensor

# Make empty arrays for sweep of fields in for-loop for code efficiency and define end for-loop
MaxAmplitude_X = np.zeros((int(h_end_step - begin_h),f_startend))
MaxAmplitude_Y = np.zeros((int(h_end_step - begin_h),f_startend))
MaxAmplitude_Z = np.zeros((int(h_end_step - begin_h),f_startend))

# define ranges that are used to put values of kittel formula in matrices
Kittel = range(f_start,f_end,10)    
Kittelarg = np.zeros(f_startend)
frequencies = range (f_start , f_end, 10)

for FreqCounter in pbar(frequencies):
    # Make empty arrays for sweep of frequencies in for-loop for code efficiency and define end for-loop
    MaxCurrent_X = np.zeros(int(h_end_step - begin_h))
    MaxCurrent_Y = np.zeros(int(h_end_step - begin_h))
    MaxCurrent_Z = np.zeros(int(h_end_step - begin_h))
    
    MinCurrent_X = np.zeros(int(h_end_step - begin_h))
    MinCurrent_Y = np.zeros(int(h_end_step - begin_h))
    MinCurrent_Z = np.zeros(int(h_end_step - begin_h))
    
    fields = range(begin_h, h_end_step)
    
    for h in fields:
        def dUdt(U, t):
            # Absorbing mx, my, mz into vector U
            mx, my, mz = U
        
            # Defining Hext vector, h/(1000*StepsPerField) --> (mT->T) and makes sure that for loop goes until end_h
            Hextx = h / (1000 * StepsPerField) * np.cos(theta_h) 
            Hexty = h / (1000 * StepsPerField) * np.sin(theta_h)
            Hextz = 0
            
            # Defining Heff vector: sum of constant term, microwave term and demagnization tensor term
            Hx = Hextx + Dx * Ms0 * mx 
            Hy = Hexty + Dy * Ms0 * my 
            Hz = Hextz + Dz * Ms0 * mz + hmicro * np.cos(FreqCounter * t)
    
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
    
        # Calculate amplitude fmr precession, A = max - min
        MaxCurrent_X[h] = max(I_charge[round(h_end_step * 0.65) : (h_end_step - 1) , 0]) 
        MaxCurrent_Y[h] = max(I_charge[round(h_end_step * 0.65) : (h_end_step - 1) , 1])
        MaxCurrent_Z[h] = max(I_charge[round(h_end_step * 0.65) : (h_end_step - 1) , 2])
        
        MinCurrent_X[h] = min(I_charge[round(h_end_step * 0.65) : (h_end_step - 1) , 0])
        MinCurrent_Y[h] = min(I_charge[round(h_end_step * 0.65) : (h_end_step - 1) , 1])
        MinCurrent_Z[h] = min(I_charge[round(h_end_step * 0.65) : (h_end_step - 1) , 2])
        
        MaxAmplitude_X[h,int((FreqCounter-f_start)/10)] = MaxCurrent_X[h] - MinCurrent_X[h]
        MaxAmplitude_Y[h,int((FreqCounter-f_start)/10)] = MaxCurrent_Y[h] - MinCurrent_Y[h]
        MaxAmplitude_Z[h,int((FreqCounter-f_start)/10)] = MaxCurrent_Z[h] - MinCurrent_Z[h]
        
        # Calculate position FMR peak and put in matrix
        Kittelarg[int((FreqCounter - f_start)/10)] = np.argmax(MaxAmplitude_Y[:,int((FreqCounter - f_start)/10)])
       
# Make array for x- and y-axis plots, from 0 to hend instead of h_end*StepsPerField, which is the range of the for-loop
x_axis = []
for x in fields:
    x_axis.append(x / StepsPerField)

y_axis = []
for y in Kittel:
    y_axis.append(y)
       
        
        
"""
PLOTTING 
"""
fig = plt.figure(figsize = plt.figaspect(.25))
fig.suptitle('Left: AC Charge Current, Right: Microwave Frequency against FMR peak')


"""
Plot Max Amplitude against the frequency
""" 
ax1 = fig.add_subplot(1 , 2 , 1)

# Grid 
ax1.grid(True)

# X-axis range
plt.xlim(0,600)

# Plotting Max Amplitudes Charge current
for i in range(0,f_startend):
    ax1.plot(x_axis, MaxAmplitude_Y[:, i], label = r'$I_y^{AC}$')

# Legend, labels, title
plt.title('Maximum AC charge current')
#ax1.legend()
ax1.set_xlabel(r'External field $\mu_0 H$ [mT]')
ax1.set_ylabel(r'Maximum amplitude [a.u.]')
#ax1.tick_params(labelleft=False) # This removes axes labels, for when you want arbitrary units

"""
Plot DI(H)/DH
"""

ax2 = fig.add_subplot(1, 2, 2)

# Grid
ax2.grid(True)

# Removing 1 frequency element because np.diff() wil be one element shorter
# If this is not done, x and y(x) dimensions will not match
x_axis_cut = x_axis[:-1]

# Y-axis range
plt.ylim(0,300)

# Plotting
plt.scatter(Kittelarg , y_axis)

# Legend, labels, title
plt.title(r'Microwave frequency against FMR peak')
#ax2.legend()
ax2.set_xlabel(r'External field $\mu_0 H$ [mT]')
ax2.set_ylabel(r'Microwave frequency $\omega$ [rad/ns]')
#ax2.tick_params(labelleft=False) 

# Saving
#plt.savefig('figuren/kittelplot.pdf' , bbox_inches = 'tight')