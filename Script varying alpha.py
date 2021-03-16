# -*- coding: utf-8 -*-
"""
Created on Mon May 21 10:30 2020

THIS SCRIPT IS USED TO MAKE FIGURES FOR CHANGING ALPHA

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
gamma = 176             # gyromagnetic ratio
mu0 = 0.4 * np.pi
theta_m = 0.2           # defines initial position magnetization vector
theta_h = 0.0 * np.pi   # angle of external magnetic field

# Gilbert damping and spin pump damping
#alpha0 = 0.05
#alphaprime = 0.05
#alpha = alpha0 + alphaprime
alpha_start = 6     # start for-loop of alpha
alpha_end = 18      # end for loop

# Microwave field
hmicro = 0.1        # amplitude
f = 120             # frequency (actually omega)

# Defining H field
begin_h = 0         # start for-loop of external field
end_h = 400         # end for-loop
StepsPerField = 1   # == end_h*steps, choose integer
h_end_step = end_h * StepsPerField  # defines end of for-loop (later, the field is divided by steps again (this to reduce step size))
Ms0 = 1.6           # Saturation demagnitization 
Dx, Dy, Dz = [0 , -0.01, -0.99] # Demagnization tensor

# Make empty arrays for sweep of fields in for-loop for code efficiency and define end for-loop
MaxAmplitude_X = np.zeros((int(h_end_step - begin_h),(alpha_end - alpha_start)))
MaxAmplitude_Y = np.zeros((int(h_end_step - begin_h),(alpha_end - alpha_start)))
MaxAmplitude_Z = np.zeros((int(h_end_step - begin_h),(alpha_end - alpha_start)))
    
# Empty array for DC field component
DC_Current_X = np.zeros((int(h_end_step - begin_h),(alpha_end - alpha_start)))
DC_Current_Y = np.zeros((int(h_end_step - begin_h),(alpha_end - alpha_start)))
DC_Current_Z = np.zeros((int(h_end_step - begin_h),(alpha_end - alpha_start)))

DIDHx = np.zeros((int(h_end_step - begin_h - 1),(alpha_end - alpha_start)))
DIDHy = np.zeros((int(h_end_step - begin_h - 1),(alpha_end - alpha_start)))
DIDHz = np.zeros((int(h_end_step - begin_h - 1),(alpha_end - alpha_start)))

for AlphaCounter in pbar(range (alpha_start,alpha_end)):
    alpha = AlphaCounter/100
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
        
        MaxAmplitude_X[h,(AlphaCounter - alpha_start)] = MaxCurrent_X[h] - MinCurrent_X[h]
        MaxAmplitude_Y[h,(AlphaCounter - alpha_start)] = MaxCurrent_Y[h] - MinCurrent_Y[h]
        MaxAmplitude_Z[h,(AlphaCounter - alpha_start)] = MaxCurrent_Z[h] - MinCurrent_Z[h]
            
        # Calculate DC charge current, which is simply the average of the charge current
        DC_Current_X[h,(AlphaCounter - alpha_start)] = np.average(I_charge[round(h_end_step*0.65) : (h_end_step-1),0])
        DC_Current_Y[h,(AlphaCounter - alpha_start)] = np.average(I_charge[round(h_end_step*0.65) : (h_end_step-1),1])
        DC_Current_Z[h,(AlphaCounter - alpha_start)] = np.average(I_charge[round(h_end_step*0.65) : (h_end_step-1),2])
       
        # Calculate DIDH of the Charge current amplitude
        DIDHx[:,(AlphaCounter - alpha_start)] = np.diff(MaxAmplitude_X[:,(AlphaCounter - alpha_start)])
        DIDHy[:,(AlphaCounter - alpha_start)] = np.diff(MaxAmplitude_Y[:,(AlphaCounter - alpha_start)])
        DIDHz[:,(AlphaCounter - alpha_start)] = np.diff(MaxAmplitude_Z[:,(AlphaCounter - alpha_start)])
        
# Make array for x-axis plots, from 0 to hend instead of h_end*StepsPerField, which is the range of the for-loop
x_axis = []
for x in fields:
    x_axis.append(x / StepsPerField)
       
        
        
"""
PLOTTING 
"""
fig = plt.figure(figsize = plt.figaspect(.25))
fig.suptitle('Left: AC Charge Current, Right: AC Charge Current $dI(H)/dH$')


"""
Plot Amplitude against the frequency
""" 
ax1 = fig.add_subplot(1 , 2 , 1)

# Grid 
ax1.grid(True)

# X- and Y- axis range
plt.xlim(0,400)
#plt.ylim(-1.5*10**(-21),1.5*10**(-21))

# Plotting Max Amplitudes Charge current
#ax1.plot(_axis, MaxAmplitude_X, label = r'$I_x^{AC}$')
ax1.plot(x_axis, MaxAmplitude_Y[:,0], label = r'$I_y^{AC}, \alpha = 0.06$')
ax1.plot(x_axis, MaxAmplitude_Y[:,1], label = r'$I_y^{AC}, \alpha = 0.07$')
ax1.plot(x_axis, MaxAmplitude_Y[:,2], label = r'$I_y^{AC}, \alpha = 0.08$')
ax1.plot(x_axis, MaxAmplitude_Y[:,3], label = r'$I_y^{AC}, \alpha = 0.09$')
ax1.plot(x_axis, MaxAmplitude_Y[:,4], label = r'$I_y^{AC}, \alpha = 0.10$')
ax1.plot(x_axis, MaxAmplitude_Y[:,5], label = r'$I_y^{AC}, \alpha = 0.11$')
ax1.plot(x_axis, MaxAmplitude_Y[:,6], label = r'$I_y^{AC}, \alpha = 0.12$')
ax1.plot(x_axis, MaxAmplitude_Y[:,7], label = r'$I_y^{AC}, \alpha = 0.13$')
ax1.plot(x_axis, MaxAmplitude_Y[:,8], label = r'$I_y^{AC}, \alpha = 0.14$')
ax1.plot(x_axis, MaxAmplitude_Y[:,9], label = r'$I_y^{AC}, \alpha = 0.15$')
ax1.plot(x_axis, MaxAmplitude_Y[:,10], label = r'$I_y^{AC}, \alpha = 0.16$')
ax1.plot(x_axis, MaxAmplitude_Y[:,11], label = r'$I_y^{AC}, \alpha = 0.17$')
#ax1.plot(x_axis, MaxAmplitude_Z, label = r'$I_z^{AC}$')

# Legend, labels, title
plt.title('AC charge current')
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

# X- and Y- axis range
plt.xlim(0, 400)
#plt.ylim(-1.5*10**(-23),1*10**(-23))

# Plotting
#ax2.plot(x_axis_cut, DIDHx, label = r'$I_x^{AC}$')
ax2.plot(x_axis_cut, DIDHy[:,0], label = r'$I_y^{AC}, \alpha = 0.06$')
ax2.plot(x_axis_cut, DIDHy[:,1], label = r'$I_y^{AC}, \alpha = 0.07$')
ax2.plot(x_axis_cut, DIDHy[:,2], label = r'$I_y^{AC}, \alpha = 0.08$')
ax2.plot(x_axis_cut, DIDHy[:,3], label = r'$I_y^{AC}, \alpha = 0.09$')
ax2.plot(x_axis_cut, DIDHy[:,4], label = r'$I_y^{AC}, \alpha = 0.10$')
ax2.plot(x_axis_cut, DIDHy[:,5], label = r'$I_y^{AC}, \alpha = 0.11$')
ax2.plot(x_axis_cut, DIDHy[:,6], label = r'$I_y^{AC}, \alpha = 0.12$')
ax2.plot(x_axis_cut, DIDHy[:,7], label = r'$I_y^{AC}, \alpha = 0.13$')
ax2.plot(x_axis_cut, DIDHy[:,8], label = r'$I_y^{AC}, \alpha = 0.14$')
ax2.plot(x_axis_cut, DIDHy[:,9], label = r'$I_y^{AC}, \alpha = 0.15$')
ax2.plot(x_axis_cut, DIDHy[:,10], label = r'$I_y^{AC}, \alpha = 0.16$')
ax2.plot(x_axis_cut, DIDHy[:,11], label = r'$I_y^{AC}, \alpha = 0.17$')
#ax2.plot(x_axis_cut, DIDHz, label = r'$I_z{AC}$')

# Legend, labels, title
plt.title(r'FMR signal $dI(H)/dH$')
#ax2.legend()
ax2.set_xlabel(r'External field $\mu_0 H$ [mT]')
ax2.set_ylabel(r'Charge $dI(H)/dH$ [a.u.]')
#ax2.tick_params(labelleft=False) 

# Saving
#plt.savefig('10juniplot.pdf' , bbox_inches = 'tight')