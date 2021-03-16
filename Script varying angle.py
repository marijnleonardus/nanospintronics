# -*- coding: utf-8 -*-
"""
Created on Mon May 21 10:30 2020

THIS SCRIPT IS USED TO MAKE FIGURES FOR CHANGING THE ANGLE OF THE MAGNETIC FIELD

@author: Roy Rosenkamp, Marijn Venderbosch
"""

# Importing packages
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from progressbar import ProgressBar
pbar = ProgressBar()
#from mpl_toolkits.mplot3d import Axes3D
# Last import registers the 3D projection, but is otherwise unused 

# Constants in units of [10^6 A/m] from mathematica notebook '3MN220 LLG model.nb'
gamma = 176                 # gyromagnetic ratio
mu0 = 0.4 * np.pi
theta_m = 0.2               # defines initial position magnetization vector
angle_start = 0             # start for-loop angle
angle_end = 6               # end for-loop angle

# defines values and ranges that are used to put result in matrices
angle_startend = int(angle_end - angle_start)
anglerange = range(angle_start,angle_end)

# Gilbert damping and spin pump damping
alpha0 = 0.05
alphaprime = 0.05
alpha = alpha0 + alphaprime

# Microwave field
hmicro = 0.1                # amplitude
f = 120                     # frequency (is omega actually)

# Defining external H field
begin_h = 0                 # start for-loop external field
end_h = 300                 # end for-loop external field
StepsPerField = 2           # == end_h*steps, choose integer
h_end_step = end_h * StepsPerField # defines end of for-loop (later, the field is divided by steps again(to reduce stepsize))
Ms0 = 1.6                   # Saturation demagnitization 
Dx, Dy, Dz = [0 , -0.01, -0.99] # Demagnization tensor

# Make empty arrays for sweep of fields in for-loop for code efficiency and define end for-loop
MaxAmplitude_X = np.zeros((int(h_end_step - begin_h),(angle_startend)))
MaxAmplitude_Y = np.zeros((int(h_end_step - begin_h),(angle_startend)))
MaxAmplitude_Z = np.zeros((int(h_end_step - begin_h),(angle_startend)))
    
# Empty array for DC field component
DC_Current_X = np.zeros((int(h_end_step - begin_h),(angle_startend)))
DC_Current_Y = np.zeros((int(h_end_step - begin_h),(angle_startend)))
DC_Current_Z = np.zeros((int(h_end_step - begin_h),(angle_startend)))

# Empty array for DIDH component
DIDHx = np.zeros((int(h_end_step - begin_h - 1),(angle_startend)))
DIDHy = np.zeros((int(h_end_step - begin_h - 1),(angle_startend)))
DIDHz = np.zeros((int(h_end_step - begin_h - 1),(angle_startend)))

for AngleCounter in pbar(anglerange):
    theta_h = AngleCounter * np.pi / 10
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
    
        # Calculate amplitude fmr precession, A = max - min
        MaxCurrent_X[h] = max(I_charge[round(h_end_step * 0.65) : (h_end_step - 1) , 0]) 
        MaxCurrent_Y[h] = max(I_charge[round(h_end_step * 0.65) : (h_end_step - 1) , 1])
        MaxCurrent_Z[h] = max(I_charge[round(h_end_step * 0.65) : (h_end_step - 1) , 2])
        
        MinCurrent_X[h] = min(I_charge[round(h_end_step * 0.65) : (h_end_step - 1) , 0])
        MinCurrent_Y[h] = min(I_charge[round(h_end_step * 0.65) : (h_end_step - 1) , 1])
        MinCurrent_Z[h] = min(I_charge[round(h_end_step * 0.65) : (h_end_step - 1) , 2])
        
        MaxAmplitude_X[h,(AngleCounter)] = MaxCurrent_X[h] - MinCurrent_X[h]
        MaxAmplitude_Y[h,(AngleCounter)] = MaxCurrent_Y[h] - MinCurrent_Y[h]
        MaxAmplitude_Z[h,(AngleCounter)] = MaxCurrent_Z[h] - MinCurrent_Z[h]
            
        # Calculate DC charge current, which is simply the average of the charge current
        DC_Current_X[h,(AngleCounter)] = np.average(I_charge[round(h_end_step*0.65) : (h_end_step-1),0])
        DC_Current_Y[h,(AngleCounter)] = np.average(I_charge[round(h_end_step*0.65) : (h_end_step-1),1])
        DC_Current_Z[h,(AngleCounter)] = np.average(I_charge[round(h_end_step*0.65) : (h_end_step-1),2])
       
        # Calculte DIDH
        DIDHx[:,(AngleCounter)] = np.diff(MaxAmplitude_X[:,(AngleCounter)])
        DIDHy[:,(AngleCounter)] = np.diff(MaxAmplitude_Y[:,(AngleCounter)])
        DIDHz[:,(AngleCounter)] = np.diff(MaxAmplitude_Z[:,(AngleCounter)])
        
# Make array for x-axis plots, from 0 to hend instead of h_end*StepsPerField, which is the range of the for-loop
x_axis = []
for x in fields:
    x_axis.append(x / StepsPerField)
       
"""
PLOTTING 
"""
fig = plt.figure(figsize = plt.figaspect(.25))
#fig.suptitle('Left: AC Charge Current, Middle: AC Charge Current $dI(H)/dH$, Right: DC Charge Current')


"""
Plot Amplitude against the frequency
""" 
ax1 = fig.add_subplot(1 , 3 , 1)

# Grid 
ax1.grid(True)

# X- and Y- axis range
plt.xlim(50,300)
#plt.ylim(-1.5*10**(-21),1.5*10**(-21))

# Plotting Max Amplitudes Charge current
#ax1.plot(_axis, MaxAmplitude_X, label = r'$I_x^{AC}$')
ax1.plot(x_axis, MaxAmplitude_Y[:,0], label = r' $\theta = 0.0 \pi$')
ax1.plot(x_axis, MaxAmplitude_Y[:,1], label = r' $\theta = 0.1 \pi$')
ax1.plot(x_axis, MaxAmplitude_Y[:,2], label = r' $\theta = 0.2 \pi$')
ax1.plot(x_axis, MaxAmplitude_Y[:,3], label = r' $\theta = 0.3 \pi$')
ax1.plot(x_axis, MaxAmplitude_Y[:,4], label = r' $\theta = 0.4 \pi$')
ax1.plot(x_axis, MaxAmplitude_Y[:,5], label = r' $\theta = 0.5 \pi$')
#ax1.plot(x_axis, MaxAmplitude_Z, label = r'$I_z^{AC}$')

# Legend, labels, title
plt.title('AC charge current $I_y^{AC}$')
ax1.legend(loc = 'upper right',title=r'$I_y^{AC}$')
ax1.set_xlabel(r'External field $\mu_0 H$ [mT]')
ax1.set_ylabel(r'Amplitude [a.u.]')
ax1.tick_params(labelleft=False) # This removes axes labels, for when you want arbitrary units

"""
Plot DI(H)/DH
"""
ax2 = fig.add_subplot(1, 3, 2)

# Grid
ax2.grid(True)

# Removing 1 frequency element because np.diff() wil be one element shorter
# If this is not done, x and y(x) dimensions will not match
x_axis_cut = x_axis[:-1]

# X- and Y- axis range
plt.xlim(50, 300)

# Plotting
#ax2.plot(x_axis_cut, DIDHx, label = r'$I_x^{AC}$')
ax2.plot(x_axis_cut, DIDHy[:,0], label = r'$\theta = 0.0 \pi$')
ax2.plot(x_axis_cut, DIDHy[:,1], label = r'$\theta = 0.1 \pi$')
ax2.plot(x_axis_cut, DIDHy[:,2], label = r'$\theta = 0.2 \pi$')
ax2.plot(x_axis_cut, DIDHy[:,3], label = r'$\theta = 0.3 \pi$')
ax2.plot(x_axis_cut, DIDHy[:,4], label = r'$\theta = 0.4 \pi$')
ax2.plot(x_axis_cut, DIDHy[:,5], label = r'$\theta = 0.5 \pi$')
#ax2.plot(x_axis_cut, DIDHz, label = r'$I_z{AC}$')

# Legend, labels, title
plt.title(r'FMR signal $dI_y(H)/dH$')
ax2.legend(loc = 'upper right', title=r'$dI_y(H)/DH$ (AC)')
ax2.set_xlabel(r'External field $\mu_0 H$ [mT]')
ax2.set_ylabel(r'FMR signal $dI(H)/dH$ [a.u.]')
#ax2.tick_params(labelleft=False) 

"""
Plot DC Current
"""
ax3 = fig.add_subplot(1, 3, 3)

# Grid
ax3.grid(True)

# Removing 1 frequency element because np.diff() wil be one element shorter
# If this is not done, x and y(x) dimensions will not match
x_axis_cut = x_axis[:-1]

# X- and Y- axis range
plt.xlim(50, 300)

# Plotting
ax3.plot(x_axis, DC_Current_X[:,0], label = r'$ \theta = 0.0 \pi$')
ax3.plot(x_axis, DC_Current_X[:,1], label = r'$\theta = 0.1 \pi$')
ax3.plot(x_axis, DC_Current_X[:,2], label = r'$ \theta = 0.2 \pi$')
ax3.plot(x_axis, DC_Current_X[:,3], label = r'$ \theta = 0.3 \pi$')
ax3.plot(x_axis, DC_Current_X[:,4], label = r'$ \theta = 0.4 \pi$')
ax3.plot(x_axis, DC_Current_X[:,5], label = r'$\theta = 0.5 \pi$')
ax3.tick_params(labelleft=False) 

# Legend, labels, title
plt.title(r'DC Charge Current $I_x^{DC}$')
ax3.legend(loc = 'upper right',title=r'$I_x^{DC}$')
ax3.set_xlabel(r'External field $\mu_0 H$ [mT]')
ax3.set_ylabel(r'DC Charge Current, $I_x^{DC}$ [a.u.]')

# Saving
#plt.savefig('HOEKplot.png', dpi=600, bbox_inches = 'tight')

