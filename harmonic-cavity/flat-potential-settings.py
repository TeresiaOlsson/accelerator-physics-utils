#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 12:05:45 2023

@author: Teresia Olsson, teresia.olsson@helmholtz-berlin.de
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import speed_of_light

# Import local modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))
from harmonic_cavity_utils import flat_potential_voltage_settings, potential, bunch_profile, bunch_length_rms, form_factor, flat_potential_cavity_settings, resonance_frequency
from cavity_input_output import read_input

#%% ========================== Read input ==============================

machine_params_file = sys.argv[1]
harm_cav_params_file = sys.argv[2]

machine_data = read_input(machine_params_file)
harm_cav_data = read_input(harm_cav_params_file)

#%% ========================== Required machine parameters from input ==============================

# revolution_time = machine_data.circumference / speed_of_light
# rf_freq = machine_data.harm_num / revolution_time

# print(rf_freq)

rf_freq = machine_data.rf_frequency
revolution_time = machine_data.harm_num / rf_freq

#%% ========================== RF acceptance ==============================

q = machine_data.rf_voltage / machine_data.energy_loss

rf_acceptance = np.sqrt( 2 * machine_data.energy_loss / (np.pi * machine_data.mom_comp * machine_data.harm_num * machine_data.energy) * ( np.sqrt(q**2 -1) - np.arcsin(1/q) ) )

print("\n")
print("RF acceptance: %5.5f %%" % (rf_acceptance*100))

#%% ========================== Natural bunch length using potential ==============================

# Synchronous phase
synchronous_phase_nat = np.pi - np.arcsin(machine_data.energy_loss / machine_data.rf_voltage)

# RF potential and longitudinal bunch profile
time_nat, potential_nat = potential(machine_data.mom_comp, rf_freq, machine_data.energy, revolution_time, machine_data.rf_voltage, synchronous_phase_nat, harm_cav_data.harmonic, 0, 0)
profile_nat = bunch_profile(time_nat, potential_nat, machine_data.mom_comp, machine_data.energy_spread)

# Center and bunch length

center_nat, bunch_length_nat = bunch_length_rms(time_nat, profile_nat)

print("Natural synchronous phase: %5.5f rad/%5.5f degree" % (synchronous_phase_nat, synchronous_phase_nat/np.pi*180))
print("Natural bunch length: %5.5f ps/%5.5f mm\n" % (bunch_length_nat*1e12, bunch_length_nat*speed_of_light*1e3))

#%% ========================== Find settings for flat potential ==============================

k_FP, harm_cav_phase_FP, synchronous_phase_FP = flat_potential_voltage_settings(machine_data.rf_voltage, machine_data.energy_loss, harm_cav_data.harmonic)

print("Flat potential settings:\n")
print("sin convention:")
print("Harmonic voltage: %5.5f keV" % (k_FP*machine_data.rf_voltage*1e-3))
print("Harmonic phase: %5.5f rad/%5.5f degree" % (harm_cav_phase_FP, harm_cav_phase_FP/np.pi*180))
print("Synchronous phase: %5.5f rad/%5.5f degree\n" % (synchronous_phase_FP, synchronous_phase_FP/np.pi*180))

print("cos convention:")
print("Harmonic voltage: %5.5f keV" % (k_FP*machine_data.rf_voltage*1e-3))
print("Harmonic phase: %5.5f rad/%5.5f degree" % (harm_cav_phase_FP-np.pi/2, (harm_cav_phase_FP-np.pi/2)/np.pi*180))
print("Synchronous phase: %5.5f rad/%5.5f degree\n" % (synchronous_phase_FP-np.pi/2, (synchronous_phase_FP-np.pi/2)/np.pi*180))

# RF potential and longitudinal bunch profile
time_FP,potential_FP = potential(machine_data.mom_comp, rf_freq,machine_data.energy, revolution_time, machine_data.rf_voltage, synchronous_phase_FP, harm_cav_data.harmonic,k_FP*machine_data.rf_voltage,harm_cav_phase_FP)
profile_FP = bunch_profile(time_FP, potential_FP, machine_data.mom_comp, machine_data.energy_spread)

# Center and bunch length
center_FP,bunch_length_FP = bunch_length_rms(time_FP, profile_FP)

print("Bunch length: %5.5f ps/%5.5f mm" % (bunch_length_FP*1e12,bunch_length_FP*speed_of_light*1e3))

# Form factor amplitude
F_amp_FP = np.absolute(form_factor(time_FP, profile_FP, harm_cav_data.harmonic, rf_freq))

harm_cav_Rs_FP, harm_cav_tuning_angle_FP = flat_potential_cavity_settings(machine_data.rf_voltage, machine_data.energy_loss, harm_cav_data.harmonic, machine_data.mom_comp, rf_freq, machine_data.energy, revolution_time, machine_data.energy_spread, machine_data.current)
harm_cav_resonance_freq_FP, harm_cav_detuning_FP = resonance_frequency(harm_cav_tuning_angle_FP, harm_cav_data.Q, harm_cav_data.harmonic, rf_freq)

print("Form factor amplitude: %5.5f" % (F_amp_FP))
print("Shunt impedance: %5.5f MOhm" % (harm_cav_Rs_FP*1e-6))
print("Tuning angle: %5.5f rad/%5.5f degree" % (harm_cav_tuning_angle_FP, harm_cav_tuning_angle_FP/np.pi*180))
print("Resonance frequency: %5.15f MHz" % (harm_cav_resonance_freq_FP*1e-6))
print("Detuning: %5.5f kHz\n" % (harm_cav_detuning_FP*1e-3))


#%% ========================== Find scalar solution ==============================
# Find the equilibrium distribution assuming a scalar form factor

# def scalar_penalty_function(F_initial,R,psi_h,I,n,Vrf,U0,alpha_c,frf,E0,T0,sigma_e):
        
#      # Calculate harmonic voltage
#      phi_h = psi_h + np.pi/2
#      VHC = -2*F_initial*R*I*np.cos(psi_h)
#      k = VHC/Vrf
#      phi_s = np.pi - np.arcsin(U0/Vrf - k*np.sin(phi_h))
    
#      # print(VHC)
#      # print(phi_h)
#      # print(phi_s)
    
#      # Calculate potential
#      time,pot = potential(alpha_c,frf,E0,T0,Vrf,phi_s,n,VHC,phi_h)
    
#      # Calculate bunch profile
#      profile = bunch_profile(time,pot,alpha_c,sigma_e)
    
#      # Calculate form factor
#      F_new = np.absolute(form_factor(time,profile,n,frf))
    
#      penalty = F_initial-F_new
    
#      return penalty

# fres = detuning + n*frf
# psi_h = np.arctan2(2*Q*detuning,fres) - np.pi # Subtracting -pi necessary to get the angle in the third quadrant

# scalar_penalty = lambda F: scalar_penalty_function(F,R,psi_h,I,n,Vrf,U0,alpha_c,frf,E0,T0,sigma_e)
# F_scalar = optimize.brentq(scalar_penalty, 0, 1)

# phi_h = psi_h + np.pi/2
# VHC = -2*F_scalar*R*I*np.cos(psi_h)
# k = VHC/Vrf
# phi_s = np.pi - np.arcsin(U0/Vrf - k*np.sin(phi_h))

# print("Scalar solution:\n")
# print("Harmonic voltage: %5.5f keV" % (k*Vrf*1e-3))
# print("Harmonic phase: %5.5f rad/%5.5f degree" % (phi_h, phi_h/np.pi*180))
# print("Tuning angle: %5.5f rad/%5.5f degree" % (psi_h, psi_h/np.pi*180))
# print("Synchronous phase: %5.15f rad/%5.15f degree\n" % (phi_s, phi_s/np.pi*180))

# time_scalar,pot_scalar = potential(alpha_c,frf,E0,T0,Vrf,phi_s,n,k*Vrf,phi_h)
# profile_scalar = bunch_profile(time_scalar,pot_scalar,alpha_c,sigma_e)
# center_scalar,bunch_length_scalar = bunch_length_rms(time_scalar,profile_scalar)

# print("Bunch length: %5.5f ps/%5.5f mm" % (bunch_length_scalar*1e12,bunch_length_scalar*c*1e3))
# print("Form factor amplitude: %5.5f\n" % (F_scalar))

#%% ========================== Find complex solution ==============================
# Find the equilibrium distribution assuming a complex form factor

# def complex_penalty_function(F_array,R,psi_h,I,n,Vrf,U0,alpha_c,frf,E0,T0,sigma_e):
    
#     F_initial = F_array[0] + 1j*F_array[1]
                
#     # Calculate harmonic voltage
#     phi_h = psi_h + np.pi/2 + np.angle(F_initial)
#     VHC = -2*np.absolute(F_initial)*R*I*np.cos(psi_h)
#     k = VHC/Vrf
#     phi_s = np.pi - np.arcsin(U0/Vrf - k*np.sin(phi_h))
        
#     # Calculate potential
#     time,pot = potential(alpha_c,frf,E0,T0,Vrf,phi_s,n,VHC,phi_h)
    
#     # Calculate bunch profile
#     profile = bunch_profile(time,pot,alpha_c,sigma_e)
    
#     # Calculate form factor
#     F_new = form_factor(time,profile,n,frf)
    
#     penalty = np.absolute(F_initial-F_new)
    
#     return penalty


# fres = detuning + n*frf
# psi_h = np.arctan2(2*Q*detuning,fres) - np.pi # Subtracting -pi necessary to get the angle in the third quadrant

# complex_penalty = lambda F_array: complex_penalty_function(F_array,R,psi_h,I,n,Vrf,U0,alpha_c,frf,E0,T0,sigma_e)

# F_complex = optimize.minimize(complex_penalty, [1,0]).x
# F_complex = F_complex[0] + 1j*F_complex[1]

# phi_h = psi_h + np.pi/2 + np.angle(F_complex)
# VHC = -2*np.absolute(F_complex)*R*I*np.cos(psi_h)
# k = VHC/Vrf
# phi_s = np.pi - np.arcsin(U0/Vrf - k*np.sin(phi_h))

# print("Complex solution:\n")
# print("Harmonic voltage: %5.5f keV" % (k*Vrf*1e-3))
# print("Harmonic phase: %5.5f rad/%5.5f degree" % (phi_h, phi_h/np.pi*180))
# print("Tuning angle: %5.5f rad/%5.5f degree" % (psi_h, psi_h/np.pi*180))
# print("Synchronous phase: %5.15f rad/%5.15f degree\n" % (phi_s, phi_s/np.pi*180))

# time_complex,pot_complex = potential(alpha_c,frf,E0,T0,Vrf,phi_s,n,k*Vrf,phi_h)
# profile_complex = bunch_profile(time_complex,pot_complex,alpha_c,sigma_e)
# center_complex,bunch_length_complex = bunch_length_rms(time_complex,profile_complex)

# print("Bunch length: %5.5f ps/%5.5f mm" % (bunch_length_complex*1e12,bunch_length_complex*c*1e3))
# print("Form factor amplitude: %5.5f" % (np.absolute(F_complex)))
# print("Form factor phase: %5.5f rad/%5.5f degree" % (np.angle(F_complex),np.angle(F_complex)/np.pi*180))

#%% ========================== Plot bunch profiles ==============================    

plt.figure(1)
plt.plot(time_FP*1e12,profile_FP,label='Theoretical')   
#plt.plot(time_scalar*1e12,profile_scalar,label='Scalar')    
#plt.plot(time_complex*1e12,profile_complex,label='Complex')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Bunch profile [arb. units]')
plt.savefig('bunch_profiles.png')



    
    







    


