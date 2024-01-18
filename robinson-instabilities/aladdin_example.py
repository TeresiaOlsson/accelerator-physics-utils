#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 10:13:10 2021

@author: Teresia Olsson, teresia.olsson@helmholtz-berlin.de
"""

"""
Example for Aladdin to reproduce results in Bosch, Kleman & Bisognano, Phys. Rev. ST. AB. 4, 074401 (2001)
"""

import numpy as np
from bosch_algorithm import *
# import sys
import matplotlib.pyplot as plt

#%% ================================ Machine parameters ================================ 

energy = 800e6 # in eV
rev_time = 2.96e-7

mom_comp = 0.0335
energy_loss = 17.4e3
energy_spread = 4.8e-4 # Relative energy spread
long_damping = 13.8e-3

machine = MachineSettings(energy, rev_time, mom_comp, energy_loss, energy_spread, long_damping)

#%% ================================ Cavity parameters ================================ 

rf_freq = 50.6e6

# Main cavity
V_MC = 90e3
R_MC = 0.5e6
Q_MC = 8000
beta_MC = 11

# Harmonic cavity
nHarm = 4
R_HC = 1.24e6
Q_HC = 20250
beta_HC = None

cavities = CavitySettings(rf_freq, V_MC, R_MC, Q_MC, beta_MC, nHarm, R_HC, Q_HC, beta_HC)

#%%% ================================  Active harmonic cavity - Robinson instabilities without coupling ================================
# Sec. V, Fig. 8 a

currents = np.linspace(0,300e-3,100)
beta_HCs = np.linspace(0,300,100)

# Assume xi = 1 and cubic term = 0 to create rougly symmetric bunch profile
xi = 1

# No coupling
coupling = False

output = []

for current in currents:   
    for beta_HC in beta_HCs:
        cavities.beta_harm = beta_HC                
        output.append( algorithm_active_cavity(machine,cavities,current,xi,coupling) )
                   
#%% Plot dipole Robinson instability

current = np.array([point['current'][0] for point in output])
betaHC = np.array([point['beta_harm'][0] for point in output])
dipole_robinson = [point['ac_robinson_exist'][0] for point in output]

index_unstable = np.argwhere(np.array(dipole_robinson) == True)
index_stable = np.argwhere(np.array(dipole_robinson) == False)
              
plt.figure(1)
plt.scatter(betaHC[index_unstable],current[index_unstable]*1e3,s=100,marker='|',label='Dipole Robinson instability')
plt.xlim([0,300])
plt.ylim([0,300])
plt.xlabel(r'$\beta_{HC}$')
plt.ylabel('Current [mA]')
plt.grid()
plt.legend()
plt.savefig('dipole_robinson_no_coupling.png',dpi=300)

#%%% ================================  Active harmonic cavity - Robinson instabilities with coupling ================================
# Sec. V, Fig. 8 b

currents = np.linspace(0,300e-3,100)
beta_HCs = np.linspace(0,300,100)

# Assume xi = 1 and cubic term = 0 to create rougly symmetric bunch profile
xi = 1

# With coupling
coupling = True

output = []

for current in currents:   
    for beta_HC in beta_HCs:
                
        cavities.beta_harm = beta_HC             
        output.append( algorithm_active_cavity(machine,cavities,current,xi,coupling) )

#%% Plot coupled-dipole

current = np.array([point['current'][0] for point in output])
betaHC = np.array([point['beta_harm'][0] for point in output])
coupled_dipole = [point['ac_robinson_exist'][0] for point in output]
fast_mode_coupling = [point['ac_robinson_exist'][2] for point in output]

index_unstable_fast = np.argwhere(np.array(fast_mode_coupling) == True)
index_stable_fast  = np.argwhere(np.array(fast_mode_coupling) == False)
        
index_unstable_coupled_dipole = np.argwhere(np.array(coupled_dipole) == True)
index_stable_coupled_dipole  = np.argwhere(np.array(coupled_dipole) == False)

plt.figure(2)
plt.scatter(betaHC[index_unstable_coupled_dipole],current[index_unstable_coupled_dipole]*1e3,s=100,marker='|',label='Coupled-dipole instability')
plt.scatter(betaHC[index_unstable_fast],current[index_unstable_fast]*1e3,s=10,marker='o',label='Fast mode-coupling instability')
plt.xlim([0,300])
plt.ylim([0,300])
plt.xlabel(r'$\beta_{HC}$')
plt.ylabel('Current [mA]')
plt.legend()
plt.grid()
plt.savefig('robinson_coupling.png',dpi=300)