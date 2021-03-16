# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 19:55:15 2020

THIS SCRIPT IS USED TO EXPORT DATA POINTS AND TO FIT THE DOUBLE LORENTZIAN

@author: Marijn Venderbosch
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from update_14_zondagavond_marijn_fitten import x_axis_cut 
from update_14_zondagavond_marijn_fitten import DIDHy

file = np.column_stack((x_axis_cut, DIDHy))

np.savetxt("exports/lorentziansfitten.csv", file, delimiter=',', header="field, DIDHy")

""" script om te fitten"""
df = pd.read_csv('exports/lorentziansfitten.csv')
df.columns = ['field' , 'DIDHy']

#remove first data point because faulty data here
df = df.iloc[1:]

field = df.field
DIDHy = df.DIDHy

plt.grid(True)
plt.plot(field, DIDHy , label='data', linestyle="", marker="o", markersize=3, linewidth=0, color='b')


def doublelorentzian(field , width, fieldFMR , I_ISHE , I_AHE):
    return I_ISHE * width**2 / ((field - fieldFMR)**2 + width**2) + I_AHE * -2 * width * (field-fieldFMR) / ((field - fieldFMR)**2 +width**2)

initial_values = [27 , 158 , 5*10**(-24), 1.6*10**(-23)]

best_values, covariance = curve_fit(doublelorentzian , field, DIDHy, p0 = initial_values)

plt.plot(field, doublelorentzian(field, *best_values), 'r-', label=' fit')
plt.legend()
plt.xlabel(r'External field strength [$\mu_0 H]$'  )
plt.ylabel(r'FMR Signal $DI(H)/DH$ [a.u.]' )
plt.tick_params(labelleft=False) 

plt.savefig('figuren/lorentizansfitten_alpha_prime_0.png', dpi = 400 , bbox_inches = 'tight')