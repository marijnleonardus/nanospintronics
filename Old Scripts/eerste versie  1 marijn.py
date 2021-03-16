# -*- coding: utf-8 -*-
"""
Created on Thu May 21 15:29:12 2020

@author: Marijn Venderbosch
"""

# importeren packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D  
#Last import registers the 3D projection, but is otherwise unused 
#so ignore message that it is unused

# constants in units of [10^6 A/m] from mathematica notebook '3MN220 LLG model.nb'
gamma = 176
mu0 = 1

# Gilbert damping
alpha = 0.1

# external field, assumed constant for now
Hx, Hy, Hz = [0.2 , 0 , 0]

""" 
Defining system of differential equations

The right hand sides are copied from the mathematica file '3MN220 LLG model.nb'
Left hand sides are time derivatives, such that a system of 3 ODEs is obtained
For ease of use the three solved variables mx, my, mz are absorbed into one vector U
"""
def dUdt(U, t):
    mx, my, mz = U
    #for RHS see mathematica file
    dmxdt = -gamma * mu0 / (1 + alpha**2) * (Hz * my - Hy * mz + alpha * ( Hy * mx * my - Hy * my**2 + Hz * mx * mz - Hx * mz**2))
    dmydt = -gamma * mu0 / (1 + alpha**2) * (Hz * mx + Hx * mz + alpha * (-Hy * mx**2 + Hx * mx * my + Hz * my * mz - Hy * mz**2))
    dmzdt = -gamma * mu0 / (1 + alpha**2) * (Hy * mx - Hx * my + alpha * (-Hz * mx**2 - Hz * my**2 + Hx * mx * mz + Hy * my * mz))

    return [dmxdt, dmydt, dmzdt]

# Create time vector. Start, stop and steps:
t_span = np.linspace(0 , 0.5 , 101)

# Initial conditions. Alignment of vector m(t=0):
mx0 = 0 
my0 = 0 
mz0 = 0.1 
Uzero = [mx0 , my0 , mz0]

# Solve system of equations DUdt for initial conditon Uzero and time range t_span
solution = odeint(dUdt, Uzero, t_span)

"""
Plotting. Two subplots next to each other. Left plot is 2D, right one is 3D
"""
# Defining figure aspect ratio and title
fig = plt.figure(figsize = plt.figaspect(.4))
fig.suptitle(r'Left: magnitization as function of time and. Right: parametric plot of $x$, $y$ and $z$ components of magnitization')

"""
Left plot
"""
# 1 by 2 subfigures, first subfigure
ax = fig.add_subplot(1 , 2 , 1)

# The 0th, 1st and 2nd columns of the vector 'solution' contain mx, my and mz respectively
ax.plot(t_span, solution[:, 0], label=r'$M_x$')
ax.plot(t_span, solution[:, 1], label=r'$M_y$')
ax.plot(t_span, solution[:, 2], label=r'$M_z$')

#turn grid and legend on
ax.grid(True)
ax.legend()

#labels
ax.set_xlabel(r'Time $t$ [ns]'); 
ax.set_ylabel(r'Magnetization $M_i/M_s, i=\{x,y,z\}$')

"""
Right plot
"""
# 1 by 2 subfigures, first subfigure
ax = fig.add_subplot(1,2, 2, projection ='3d')
ax.plot(solution[:,0],solution[:,1],solution[:,2], label= "Parametric plot")

#legend
ax.legend()

#labels
ax.set_xlabel(r'$M_x/M_s$')
ax.set_ylabel(r'$M_y/M_s$')
ax.set_zlabel(r'$M_z/M_s$')


# save result
plt.savefig('2D_and_3D_plot.pdf',bbox_inches='tight')

