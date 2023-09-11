#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 12:06:32 2023

@author: Teresia Olsson, teresia.olsson@helmholtz-berlin.de
"""

import numpy as np

def flat_potential_voltage_settings(Vrf,U0,n):
    
    k = np.sqrt(1/n**2-1/(n**2-1)*(U0/Vrf)**2)
    
    phi_h = np.arctan(- n*U0/Vrf/np.sqrt((n**2-1)**2-(n**2*U0/Vrf)**2))
    
    phi_s = np.pi - np.arcsin(n**2/(n**2-1)*U0/Vrf)
    
    return k,phi_h,phi_s

def potential(alpha_c,frf,E0,T0,Vrf,phi_s,n,V_h,phi_h):
        
    omega_rf = 2*np.pi*frf
    k = V_h/Vrf
    
    phi = np.linspace(-np.pi/2,np.pi/2,10001)
    time = phi/omega_rf    
    
    potential = - alpha_c/(omega_rf*E0*T0)*Vrf*(np.cos(phi_s)-np.cos(phi+phi_s) + k/n*(np.cos(phi_h)-np.cos(n*phi+phi_h)) - phi*(np.sin(phi_s)+k*np.sin(phi_h)))
        
    return time,potential
    
def bunch_profile(time,potential,alpha_c,sigma_e):
        
    profile = np.exp(-potential/(alpha_c**2*sigma_e**2))
    area = np.trapz(profile,time)
    profile = profile/area
    
    return profile

def bunch_length_rms(pos,profile):
        
    # Mean of probability distribution
    aux = profile*pos
    center = np.sum(aux)/np.sum(profile)
    
    # Standard deviation of probability distribution
    aux = profile*(pos-center)**2 
    sigma  = np.sqrt(np.sum(aux)/np.sum(profile))	

    return center,sigma    

def form_factor(time,profile,n,frf):
    
    Fharm = np.trapz(profile*np.exp(-1j*2*np.pi*n*frf*time),time)
    F0 = np.trapz(profile,time)
    F = np.divide(Fharm,F0)
    
    return F

def flat_potential_cavity_settings(Vrf,U0,n,alpha_c,frf,E0,T0,sigma_e,I):
    
    k,phi_h,phi_s = flat_potential_voltage_settings(Vrf,U0,n)
    
    time,pot = potential(alpha_c,frf,E0,T0,Vrf,phi_s,n,k*Vrf,phi_h)
    profile = bunch_profile(time,pot,alpha_c,sigma_e)
    center,bunch_length = bunch_length_rms(time,profile)
    F_amp = np.absolute(form_factor(time,profile,n,frf))
    
    psi_h = phi_h - np.pi/2
    Rs = - k*Vrf/(2*F_amp*I*np.cos(psi_h))
    
    return Rs,psi_h

def resonance_frequency(tuning_angle,Q,n,frf):
    
    fres = 2*Q*n*frf/(2*Q-np.tan(tuning_angle))
    detuning = fres - n*frf
    
    return fres,detuning

    


    
    
    