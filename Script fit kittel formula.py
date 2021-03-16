# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 22:42:36 2020

THIS SCRIPT IS USED TO MAKE THE FIGURE WHILE FITTING THE KITTEL FORMULA

@author: Marijn Venderbosch

"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

df = pd.read_csv('exports/kittel.csv' , sep=';')

df.columns= ['field' , 'frequency']

fields = df.field
frequency = df.frequency

#plot data

plt.grid()
plt.scatter(fields , frequency, color='b', label='Simulation data')

#optimizing
def kittel(field, magn, prefactor):
    return prefactor * np.sqrt(field * (field + magn))

init_values = [1, -10]

best_values , coraviances = curve_fit(kittel, fields, frequency, p0=init_values)

#plot fit
plt.plot(fields, kittel(fields, *best_values), color='r', label=' fit')

# Legend, labels, title
plt.title(r'Microwave frequency against FMR peak')
plt.legend()
plt.xlabel(r'External field $\mu_0 H$ [mT]')
plt.ylabel(r'Microwave frequency $\omega$ [rad/ns]')

#save fig
plt.savefig('figuren/kittelfitje,png', dpi=400 , bbox_inches = 'tight')