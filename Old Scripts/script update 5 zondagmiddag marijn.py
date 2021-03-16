# -*- coding: utf-8 -*-
"""
Created on Thu May 21 15:29:12 2020

@author: Marijn Venderbosch, Roy Rosenkamp
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

# Gilbert damping
alpha = 0.05

# Microwave field
h = 0.1
omega = 35.2

# Defining H field
# Saturation demagnitization 
Ms0 = 1.6
# Constant external field 
Hextx, Hexty, Hextz = [0.2 , 0 , 0]
# Demagnization tensor
Dx, Dy, Dz = [0 , -0.01, -0.99]


""" 
Defining system of differential equations

The right hand sides are copied from the mathematica file '3MN220 LLG model.nb'
Left hand sides are time derivatives, such that a system of 3 ODEs is obtained
For ease of use the three solved variables mx, my, mz are absorbed into one vector U
"""

def dUdt(U, t):
    # Absorbing mx, my, mz into vector U
    mx, my, mz = U
    
    # Defining Heff vector
    # Addition of constant term, microwave term and demagnization tensor term
    Hx = Hextx + Dx * Ms0 * mx 
    Hy = Hexty + Dy * Ms0 * my 
    Hz = Hextz + Dz * Ms0 * mz + h * np.cos(omega * t)

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
I_current = D_ISHE * np.cross(I_spin, sigma)


"""
Plotting. Two subplots next to each other. Left plot is 2D, right one is 3D
"""
# Defining figure aspect ratio and title
fig = plt.figure(figsize = plt.figaspect(.25))
fig.suptitle(r'Several timedependent and parametric plots')

"""
Left plot
"""
# 1 by 2 subfigures, first subfigure
ax = fig.add_subplot(1 , 3 , 1)

# The 0th, 1st and 2nd columns of the vector 'solution' contain mx, my and mz respectively
ax.plot(t_span, magnetization[:, 0], label = r'$M_x$')
ax.plot(t_span, magnetization[:, 1], label = r'$M_y$')
ax.plot(t_span, magnetization[:, 2], label = r'$M_z$')

# Enable grid and legend
ax.grid(True)
ax.legend()

# :abels
ax.set_xlabel(r'Time $t$ [ns]'); 
ax.set_ylabel(r'Magnetization $M_i/M_s, i=\{x,y,z\}$')

"""
Middle plot
"""
# 1 by 2 subfigures, first subfigure
ax = fig.add_subplot(1 , 3 , 2 , projection = '3d')
ax.plot(magnetization[:,0] , magnetization[:,1] , magnetization[:,2] , label = "Parametric plot")

# Enable legend
ax.legend()

# Labels
ax.set_xlabel(r'$M_x/M_s$')
ax.set_ylabel(r'$M_y/M_s$')
ax.set_zlabel(r'$M_z/M_s$')

"""
Right plot
"""

ax = fig.add_subplot(1 , 3 , 3)

# All component of the spin current
#ax.plot(t_span, I_spin[:, 0], label = r'$I_spin^X$')
ax.plot(t_span, I_spin[:, 1], label = r'$I_{spin}^Y$')
#ax.plot(t_span, I_spin[:, 2], label = r'$I_spin^Z$')

# Legend and grid
ax.grid(True)
ax.legend()

# Labels
ax.set_xlabel(r'Time $t$ [ns]')
ax.set_ylabel(r'Current $I_{spin}$ [A]')


# Save result
#plt.savefig('figuren/zondagmiddag.pdf' , bbox_inches = 'tight')

