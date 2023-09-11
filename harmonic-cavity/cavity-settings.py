#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 10:49:28 2021

@author: khw79751
"""

import sys
from types import SimpleNamespace  
import numpy as np
import matplotlib.pyplot as plt

""" Read in parameters from input file """
def read_input(filename):
    
    # Open file
    f = open(filename)
    
    # Read parameters into dict
    params={}
    for line in f:
        if line[0] != '#':
            values=line.split()
            if values[1] == "optimal":
                params[values[0]] = "optimal"
            else:
                params[values[0]] = float(values[1])
    
    # Close file
    f.close()
    
    return params
    
# def Read_data(ftwiss, facc, param, HC):

# 	# read twiss file
# 	tfile = open(ftwiss, 'rU')
# 	twiss = numpy.zeros(12)
# 	newrow = numpy.zeros(12)
# 	element = numpy.zeros(1)
# 	for aline in tfile:
# 		values = aline.split()
# 		if (values[0] == '@'):
# 			if (values[1] == 'K_beta'):
# 				param['k_beta'] = float(values[3])
# 			elif (values[1] == 'K_dw'):
# 				param['k_dw'] = float(values[3])-param['k_beta']
# 			elif (values[1] == 'EX'):
# 				param['ex0'] = float(values[3])
# 		elif (values[0] <> '@' and values[0] <> '*'):
# 			if float(values[3]) > 0:
# 				for i in range(10):
# 					newrow[i] = float(values[i+2])
# 				twiss = numpy.vstack((twiss, newrow))
# 				element = numpy.vstack((element, values[0]))
# 	tfile.close()

# %% --------------------------------------------------------------------- Load input parameters ------------------------------------------------------------------------------

# Get filename from command line arguments
#filename = sys.argv[1]

filename = "/home/teresia/Documents/git-repos/accelerator-physics-utils/harmonic-cavity/cavity-settings/examples/cavity-input-diamond-ii-bare.json"

# Read in data from file
data = read_input(filename)

# Turn data into namespace to easier access the parameters
data = SimpleNamespace(**data)

# %% --------------------------------------------------------------------- Calculate flat potential settings ------------------------------------------------------------------------------





#%% --------------------------------------------------------------------- Potential, bunch profile and form factors -----------------------------------------------------

# Synchronous phase

k = data.V_HC/data.V_MC
#data.phi_MC = np.pi - np.arcsin((data.U0 - k*data.V_MC*np.sin(data.phi_HC))/data.V_MC)

# Total voltage

time = np.linspace(-200e-12,200e-12,1000)
omega_rf = 2*np.pi*frf
phi = omega_rf*time

total_voltage = V_MC*np.sin(phi+phi_MC) + V_HC*np.sin(nHarm*phi+phi_HC)

plt.figure(1)
plt.plot(phi,total_voltage*1e-6)
plt.xlabel('Phase offset [rad]')
plt.ylabel('Total voltage [MV]')
plt.grid()

# Potential

c = 299792458
C = h*c/frf

z = c*time

potential = -(alpha/(E0*C)*c*V_MC/omega_rf*( np.cos(phi_MC)-np.cos(omega_rf/c*z+phi_MC) + k/nHarm*(np.cos(phi_HC)-np.cos(nHarm*omega_rf/c*z+phi_HC)) ) -  alpha/(E0*C)*U0*z)

profile = np.exp(-potential/(alpha**2*sigma_e**2))
area = np.trapz(profile,time)
profile = profile/area

plt.figure(2)
plt.plot(time,potential)

plt.figure(3)
plt.plot(time,profile)

# Form factors

F0 = np.trapz(profile,time)
F_MC = np.trapz(profile*np.exp(-1j*2*np.pi*frf*time),time)
F_HC = np.trapz(profile*np.exp(-1j*2*np.pi*nHarm*frf*time),time)

# Only use amplitude
F_MC = np.absolute(F_MC)
F_HC = np.absolute(F_HC)

#%% --------------------------------------------------------------------- Optimal power coupling -----------------------------------------------------

V_MC_per_cav = V_MC/n_MC
V_HC_per_cav = V_HC/n_HC

psi_MC = phi_MC - np.pi/2
psi_HC = phi_HC - np.pi/2

optimal_beta_MC = 1 + 2*F_MC*I*R_MC*np.cos(psi_MC)/V_MC_per_cav
optimal_beta_HC = 1 + 2*F_HC*I*R_HC*np.cos(psi_HC)/V_HC_per_cav

#%% --------------------------------------------------------------------- Optimal detuning -----------------------------------------------------

if beta_MC == 'optimal':
    beta_MC = optimal_beta_MC

optimal_psi_MC = np.arctan(-2*F_MC*R_MC/(1+beta_MC)*I/V_MC_per_cav*np.sin(psi_MC))
optimal_fres_MC = 2*Q_MC/(1+beta_MC)*frf/(2*Q_MC/(1+beta_MC)-np.tan(optimal_psi_MC))
optimal_detuning_MC = optimal_fres_MC - frf 

if detuning_MC == 'optimal':
    psi_MC = optimal_psi_MC
    fres_MC = optimal_fres_MC
    detuning_MC = optimal_detuning_MC 
else:
    fres_MC = frf + detuning_MC
    psiMC = 2*Q_MC/(1+beta_MC)*(fres_MC-frf)/fres_MC
    
if beta_HC == 'optimal':
    beta_HC = optimal_beta_HC

optimal_psi_HC = np.arctan(-2*F_HC*R_HC/(1+beta_HC)*I/V_HC_per_cav*np.sin(psi_HC))
optimal_fres_HC = 2*Q_HC/(1+beta_HC)*nHarm*frf/(2*Q_HC/(1+beta_HC)-np.tan(optimal_psi_HC))
optimal_detuning_HC = optimal_fres_HC - nHarm*frf 

if detuning_HC == 'optimal':
    psi_HC = optimal_psi_HC
    fres_HC = optimal_fres_HC
    detuning_HC = optimal_detuning_HC 
else:
    fres_HC = nHarm*frf + detuning_HC
    psiHC = 2*Q_HC/(1+beta_HC)*(fres_HC-frf)/fres_HC    

#%% --------------------------------------------------------------------- Print results -----------------------------------------------------
print('\n')
print('Current [mA] = {0:.15f}'.format(I*1e3))

print('\n')
print('Form factor amplitudes: \n')
print('MC: {0:.15f}\nHC: {1:.15f}'.format(F_MC,F_HC))
    
print('\n')
print('Main cavity set points:\n')
print('Number of cavities = {0:d}'.format(n_MC))
print('Drive frequency [MHz] = {0:.15f}'.format(frf*1e-6))
print('Voltage [kV] = {0:.15f}'.format(V_MC_per_cav*1e-3))
print('Phase [degree] = {0:.15f}'.format(phi_MC/np.pi*180))
print('Optimal beta = {0:.15f}'.format(optimal_beta_MC))
print('Optimal detuning [kHz] = {0:.15f}'.format(optimal_detuning_MC*1e-3))
print('Beta = {0:.15f}'.format(beta_MC))
print('Tuning angle [degree] = {0:.15f}'.format(psi_MC/np.pi*180))
print('Detuning [kHz] = {0:.15f}'.format(detuning_MC*1e-3))
print('Resonance frequency [MHz] = {0:.15f}'.format(fres_MC*1e-6))

print('\n')
print('Harmonic cavity set points:\n')
print('Number of cavities = {0:d}'.format(n_HC))
print('Drive frequency [MHz] = {0:.15f}'.format(nHarm*frf*1e-6))
print('Voltage [kV] = {0:.15f}'.format(V_HC_per_cav*1e-3))
print('Phase [degree] = {0:.15f}'.format(phi_HC/np.pi*180))
print('Optimal beta = {0:.15f}'.format(optimal_beta_HC))
print('Optimal detuning [kHz] = {0:.15f}'.format(optimal_detuning_HC*1e-3))
print('Beta = {0:.15f}'.format(beta_HC))
print('Tuning angle [degree] = {0:.15f}'.format(psi_HC/np.pi*180))
print('Detuning [kHz] = {0:.15f}'.format(detuning_HC*1e-3))
print('Resonance frequency [MHz] = {0:.15f}'.format(fres_HC*1e-6))
