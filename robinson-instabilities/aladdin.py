#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 10:13:10 2021

@author: khw79751
"""
import numpy as np
from harmonic_cavity_stability_analysis import *
import sys
import matplotlib.pyplot as plt

#%% ================================ Machine parameters ================================ 

E = 800e6 # in eV
T0 = 2.96e-7

alpha = 0.0335
sigma_e = 4.8e-4 # Relative energy spread
frf = 50.6e6
U0 = 17.4e3
tauL = 13.8e-3

#%% ================================ Cavity parameters ================================ 

VMC = 90e3
RMC = 0.5e6
QMC = 8000
betaMC = 11

nHarm = 4
RHC = 1.24e6
QHC = 20250


#%%% ================================  Active harmonic cavity - Robinson instabilities without coupling ================================

#I = np.linspace(0,300e-3,100)
#betaHC = np.linspace(0,300,100)
#
## Assume xi = 1 and cubic term = 0 to create rougly symmetric bunch profile
#xi = 1
#
#equilibrium_phase = np.zeros((len(betaHC),len(I)))
#DC_robinson_exists = np.zeros((len(betaHC),len(I)))
#dipole_robinson_exists = np.zeros((len(betaHC),len(I)))
#quadrupole_robinson_exists = np.zeros((len(betaHC),len(I)))
#sextupole_robinson_exists = np.zeros((len(betaHC),len(I)))
#octupole_robinson_exists = np.zeros((len(betaHC),len(I)))
#
#for i in range(len(betaHC)):
#    for j in range(len(I)):
#
#        # Find harmonic cavity settings
#        psiMC,psiHC = phases_symmetric_profile(VMC,nHarm,xi,U0)
#        VHC = - xi*VMC*np.sin(psiMC)/(nHarm*np.sin(psiHC))
#        
#        # Check so equilibrium phase exist
#        if equilibrium_phase_exists(psiMC):
#            equilibrium_phase[i,j] = 1
#        else:
#            continue  
#            
#        # Coefficients of the synchrotron potential and plot potential
#        a,b,c = potential_coefficients(E,T0,alpha,frf,VMC,psiMC,nHarm,VHC,psiHC)
#                
#        # Synchrotron frequency
##        omega_s = synchrotron_frequency(a)
#                
#        # Bunch length
#        sigma_t = bunch_length(alpha,sigma_e,a,b,c,10000e-12) # How should be the range be chosen?
#                
#        # Form factors
#        FMC,FHC = form_factors(frf,nHarm,sigma_t)
#        
#        # Loaded cavity parameters
#        RMC_L = RMC/(1+betaMC)
#        QMC_L = QMC/(1+betaMC)
#
#        RHC_L = RHC/(1+betaHC[i])
#        QHC_L = QHC/(1+betaHC[i])
#
#        # Cavity tunings
#        phiMC = compensated_condition(VMC,psiMC,FMC,RMC_L,I[j])
#        phiHC = compensated_condition(VHC,psiHC,FHC,RHC_L,I[j])
#        
#        DC_robinson_exists[i,j],dipole_robinson_exists[i,j],quadrupole_robinson_exists[i,j],sextupole_robinson_exists[i,j],octupole_robinson_exists[i,j] = robinson_no_coupling(E,T0,alpha,sigma_e,frf,VMC,psiMC,xi,nHarm,sigma_t,FMC,FHC,RMC_L,QMC_L,phiMC,RHC_L,QHC_L,phiHC,I[j],b,c,tauL)
#                    
##%% Plot dipole Robinson instability
#        
#dipole_robinson_flatten = dipole_robinson_exists.flatten()
#
#I_flatten = np.tile(I,len(betaHC))
#betaHC_flatten = np.repeat(betaHC,len(I))        
#        
#index_unstable = np.argwhere(dipole_robinson_flatten == 1)
#index_stable = np.argwhere(dipole_robinson_flatten == 0)
#               
#plt.figure(2)
#plt.scatter(betaHC_flatten[index_unstable],I_flatten[index_unstable]*1e3,s=100,marker='|',label='Dipole Robinson instability')
#plt.xlim([0,300])
#plt.ylim([0,300])
#plt.xlabel(r'$\beta_{HC}$')
#plt.ylabel('Current [mA]')
#plt.grid()
#plt.legend()
#plt.savefig('dipole_robinson_no_coupling.png',dpi=300)

#%%% ================================  Active harmonic cavity - Robinson instabilities with coupling ================================

I = np.linspace(0,300e-3,100)
betaHC = np.linspace(0,300,100)

# Assume xi = 1 and cubic term = 0 to create rougly symmetric bunch profile
xi = 1

equilibrium_phase = np.zeros((len(betaHC),len(I)))
equilibrium_phase_instability = np.zeros((len(betaHC),len(I)))
DC_robinson_exists = np.zeros((len(betaHC),len(I)))
fast_mode_coupling_exists = np.zeros((len(betaHC),len(I)))
coupled_dipole_exists = np.zeros((len(betaHC),len(I)))
coupled_quadrupole_exists = np.zeros((len(betaHC),len(I)))

Omega = np.zeros((len(betaHC),len(I)))

for i in range(len(betaHC)):
    for j in range(len(I)):

        # Find harmonic cavity settings
        psiMC,psiHC = phases_symmetric_profile(VMC,nHarm,xi,U0)
        VHC = - xi*VMC*np.sin(psiMC)/(nHarm*np.sin(psiHC))
        
        # Check so equilibrium phase exist
        if equilibrium_phase_exists(psiMC):
            equilibrium_phase[i,j] = 1
        else:
            continue  
            
        # Coefficients of the synchrotron potential and plot potential
        a,b,c = potential_coefficients(E,T0,alpha,frf,VMC,psiMC,nHarm,VHC,psiHC)
                
        # Synchrotron frequency
#        omega_s = synchrotron_frequency(a)
                
        # Bunch length
        sigma_t = bunch_length(alpha,sigma_e,a,b,c,10000e-12) # How should be the range be chosen?
                
        # Form factors
        FMC,FHC = form_factors(frf,nHarm,sigma_t)
        
        # Loaded cavity parameters
        RMC_L = RMC/(1+betaMC)
        QMC_L = QMC/(1+betaMC)

        RHC_L = RHC/(1+betaHC[i])
        QHC_L = QHC/(1+betaHC[i])

        # Cavity tunings
        phiMC = compensated_condition(VMC,psiMC,FMC,RMC_L,I[j])
        phiHC = compensated_condition(VHC,psiHC,FHC,RHC_L,I[j])
        
        Omega[i,j],DC_robinson_exists[i,j],fast_mode_coupling_exists[i,j],coupled_dipole_exists[i,j],coupled_quadrupole_exists[i,j] = robinson_coupling(E,T0,alpha,sigma_e,frf,VMC,psiMC,xi,nHarm,sigma_t,FMC,FHC,RMC_L,QMC_L,phiMC,RHC_L,QHC_L,phiHC,I[j],b,c,tauL)


#%% Plot coupled-dipole

fast_mode_coupling_flatten = fast_mode_coupling_exists.flatten()     
coupled_dipole_flatten = coupled_dipole_exists.flatten()

I_flatten = np.tile(I,len(betaHC))
betaHC_flatten = np.repeat(betaHC,len(I))        

index_unstable_fast = np.argwhere(fast_mode_coupling_flatten == 1)
index_stable_fast  = np.argwhere(fast_mode_coupling_flatten == 0)
        
index_unstable_coupled_dipole = np.argwhere(coupled_dipole_flatten == 1)
index_stable_coupled_dipole  = np.argwhere(coupled_dipole_flatten == 0)
               
plt.figure(2)
plt.scatter(betaHC_flatten[index_unstable_coupled_dipole],I_flatten[index_unstable_coupled_dipole]*1e3,s=100,marker='|',label='Coupled-dipole instability')
plt.scatter(betaHC_flatten[index_unstable_fast],I_flatten[index_unstable_fast]*1e3,s=10,marker='o',label='Fast mode-coupling instability')
plt.xlim([0,300])
plt.ylim([0,300])
plt.xlabel(r'$\beta_{HC}$')
plt.ylabel('Current [mA]')
plt.legend()
plt.grid()
plt.savefig('robinson_coupling.png',dpi=300)

Omega_flatten = Omega.flatten()
index_nan = np.isnan(Omega.flatten())

plt.figure(3)
plt.imshow(Omega)
plt.colorbar()
#plt.scatter(betaHC_flatten[index_nan],I_flatten[index_nan])
#plt.scatter(betaHC_flatten[index_unstable_fast],I_flatten[index_unstable_fast]*1e3,s=10,marker='o',label='Fast mode-coupling instability')
#plt.xlim([0,30])
#plt.ylim([0,0.3])

#%%% ================================  Active harmonic cavity - Dipole coupled-bunch instability ================================
    
#betaHC = 160
#I = np.linspace(0,300e-3,100)
#xi = np.linspace(0.001,2,100)  
#
#equilibrium_phase = np.zeros((len(xi),len(I)))    
#CB_dipole_instability_exists = np.zeros((len(xi),len(I)))
#
#for i in range(len(xi)):
#    for j in range(len(I)):
#                
#        # Find harmonic cavity settings
#        psiMC,psiHC = phases_symmetric_profile(VMC,nHarm,xi[i],U0)
#        VHC = - xi[i]*VMC*np.sin(psiMC)/(nHarm*np.sin(psiHC))
#        
#        # Check so equilibrium phase exist
#        if equilibrium_phase_exists(psiMC):
#            equilibrium_phase[i,j] = 1
#        else:
#            continue  
#
#        # Coefficients of the synchrotron potential and plot potential
#        a,b,c = potential_coefficients(E,T0,alpha,frf,VMC,psiMC,nHarm,VHC,psiHC)
#        
##        # Synchrotron frequency
##        omega_s = synchrotron_frequency(a)
#
#        # Bunch length
#        sigma_t = bunch_length(alpha,sigma_e,a,b,c,10000e-12) # How should be the range be chosen?
#
#        # Form factors
#        FMC,FHC = form_factors(frf,nHarm,sigma_t)
#
#        # Loaded cavity parameters
#        RMC_L = RMC/(1+betaMC)
#        QMC_L = QMC/(1+betaMC)
#
#        RHC_L = RHC/(1+betaHC)
#        QHC_L = QHC/(1+betaHC)
#
#        # Cavity tunings
#        phiMC = compensated_condition(VMC,psiMC,FMC,RMC_L,I[j])
#        phiHC = compensated_condition(VHC,psiHC,FHC,RHC_L,I[j])
#    
#        CB_dipole_instability_exists[i,j] = dipole_coupled_bunch(E,T0,alpha,sigma_e,frf,VMC,psiMC,xi[i],nHarm,sigma_t,FMC,FHC,RMC_L,QMC_L,phiMC,RHC_L,QHC_L,phiHC,I[j],b,c,tauL)
#
##%% Plot dipole coupled-bunch
#
#CB_dipole_flatten = CB_dipole_instability_exists.flatten()     
#
#I_flatten = np.tile(I,len(xi))
#xi_flatten = np.repeat(xi,len(I))        
#
#index_unstable = np.argwhere(CB_dipole_flatten == 1)
#index_stable = np.argwhere(CB_dipole_flatten == 0)
#        
#               
#plt.figure(3)
#plt.scatter(xi_flatten[index_unstable],I_flatten[index_unstable]*1e3,s=10,marker='o',label='Dipole coupled-bunch instability')
#plt.xlim([0,2])
#plt.ylim([0,300])
#plt.xlabel(r'$\xi$')
#plt.ylabel('Current [mA]')
#plt.legend()
#plt.grid()
#plt.savefig('dipole_CB.png',dpi=300)