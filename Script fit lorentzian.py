# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:55:11 2020

THIS SCRIPT IS USED TO FIT THE DOUBLE LORENTZIAN

@author: Marijn Venderbosch
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

""" script om te fitten"""
df = pd.read_csv('exports/lorentziansfitten_alphaprime0.csv')
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

plt.plot(field, doublelorentzian(field, *best_values), 'r-', label=' fit', linewidth=4)
plt.legend()
plt.xlabel(r'External field strength [$\mu_0 H]$'  )
plt.ylabel(r'FMR Signal $DI(H)/DH$ [a.u.]' )
plt.tick_params(labelleft=False) 

plt.savefig('figuren/lorentziansFitten/lorentizansfitten_alpha_prime_0.png', dpi = 400 , bbox_inches = 'tight')