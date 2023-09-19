#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 10:13:10 2021

@author: khw79751
"""
import numpy as np
import scipy.integrate as integrate
import math
import scipy.misc as misc
import matplotlib.pyplot as plt

e = 1.60217662e-19

def phases_symmetric_profile(VT1,nu,xi,Vs):
    
    '''Calculate synchronous phase angles and harmonic cavity voltage based on assuming cubic term to be zero = approximately symmetric bunch'''
       
    psi1 = np.arccos(Vs/(1-1/nu**2)/VT1)    
    psi2 = np.arctan(nu*xi*np.tan(psi1)) - np.pi # Why - pi?
    
    return psi1,psi2

def equilibrium_phase_exists(psi1):
    '''Check if equilibrium phase exist'''
    
    # If synchronous phase not between 0 and 90 degress no phase focusing
    if psi1 < 0 or psi1 > np.pi/2:
        return False
    else:
        return True
    
def optimal_beta(I,R,V,psi):
    '''Calculate optimal power coupling'''
    
    beta = 1 + 2*I*R*np.cos(psi)/V
    return beta
    
def potential_coefficients(E,T0,alpha,frf,VT1,psi1,nu,VT2,psi2):
    
    omega_g = 2*np.pi*frf
    E = E*e # Change to SI units
    
    a = alpha*e*omega_g/(2*E*T0)*(VT1*np.sin(psi1)+nu*VT2*np.sin(psi2))
    b = alpha*e*omega_g**2/(6*E*T0)*(VT1*np.cos(psi1)+nu**2*VT2*np.cos(psi2))
    c = -alpha*e*omega_g**3/(24*E*T0)*(VT1*np.sin(psi1)+nu**3*VT2*np.sin(psi2))
           
    return a,b,c

def synchrotron_frequency(a):
    
    omega_s = np.sqrt(2*a)
    
    return omega_s

def bunch_length(alpha,sigma_e,a,b,c,tau):
    
    # Filling height
    #U0 = alpha**2*(sigma_e/E)**2/2
    U0 = alpha**2*sigma_e**2/2
    
    U = lambda tau: a*tau**2 + b*tau**3 + c*tau**4

    numerator = lambda tau: tau**2*np.exp(-U(tau)/(2*U0))
    denominator = lambda tau: np.exp(-U(tau)/(2.*U0))

    sigma_t = np.sqrt(integrate.quad(numerator,-tau,tau)[0]/integrate.quad(denominator,-tau,tau)[0])
    
    return sigma_t

def form_factors(frf,nu,sigma_t):
    
    omega_g = 2*np.pi*frf
    
    F1 = np.exp(-omega_g**2*sigma_t**2/2)
    F2 = np.exp(-nu**2*omega_g**2*sigma_t**2/2)
    
    return F1,F2

def compensated_condition(VT,psi,F,R,I):
    
    phi = np.arctan(2*F*I*R*np.sin(psi)/VT)
    
    return phi

    
def robinson_no_coupling(E,T0,alpha,sigma_e,frf,VT1,psi1,xi,nu,sigma_t,F1,F2,R1,Q1,phi1,R2,Q2,phi2,I,b,c,tauL):
    
    # Initilise return parameters
    zero_frequency = np.nan
    dipole_robinson =  np.nan
    quadrupole_robinson =  np.nan
    quadrupole_robinson =  np.nan
    quadrupole_robinson =  np.nan
           
    omega_g = 2*np.pi*frf
    E = E*e # Change to SI units since that is used in formulas
    
    omega1 = 2*Q1*omega_g/(2*Q1+np.tan(phi1))
    omega2 = 2*Q2*nu*omega_g/(2*Q2+np.tan(phi2))
                
    # Robinson frequency
    omega_r = np.sqrt(alpha*e*omega_g/(E*T0)*(F1-xi*F2)*VT1*np.sin(psi1))
                   
    # Zero frequency instability    
    zero_frequency_formula = lambda mu: alpha*e*omega_g*I/(E*T0)*(omega_g*sigma_t)**(2*mu-2)/(2**(mu-1)*math.factorial(mu))*(R1*F1**2*np.sin(2*phi1) + nu**(2*mu-1)*R2*F2**2*np.sin(2*phi2))
           
    if omega_r**2 < zero_frequency_formula(1):
        zero_frequency = 1        
    elif omega_r**2 < zero_frequency_formula(2):
        zero_frequency = 2
    elif omega_r**2 < zero_frequency_formula(3):        
        zero_frequency = 3        
    elif omega_r**2 < zero_frequency_formula(4):
        zero_frequency = 4
    else:
        zero_frequency = 0
        
    # If zero frequency instability exists return without calculating AC instability        
    if zero_frequency != 0:        
        return zero_frequency, dipole_robinson,quadrupole_robinson,quadrupole_robinson,quadrupole_robinson
                        
    # Landau threshold
    dipole_landau = 0.78*alpha**2*sigma_e**2/omega_r*np.absolute(3*c/omega_r**2 - (2*b/omega_r**2)**2)
        
    landau_threshold = np.zeros((4,1))
    landau_threshold[0] = dipole_landau
    landau_threshold[1] = 2.24/0.78*dipole_landau
    landau_threshold[2] = 4.12/0.78*dipole_landau
    landau_threshold[3] = 6.36/0.78*dipole_landau
            
    # AC robinson growth rates  
    Omega = np.zeros((4,1))
    alpha_r = np.zeros((4,1))
    alpha_r_incl_rad_damp = np.zeros((4,1))
    delta_Omega = np.zeros((4,1),dtype=complex)
            
    for i in [1,2,3,4]:
        
        mu = i
                            
        # Initial guess based on zero current
        Omega0 = mu*omega_r
                
        # Solve using Newton-Raphson method
        
        phi1_pos = lambda Omega: np.arctan(2*Q1*(omega_g+Omega-omega1)/omega1)
        phi1_neg = lambda Omega: np.arctan(2*Q1*(omega_g-Omega-omega1)/omega1)
                                    
        phi2_pos = lambda Omega: np.arctan(2*Q2*(nu*omega_g+Omega-omega2)/omega2)
        phi2_neg = lambda Omega: np.arctan(2*Q2*(nu*omega_g-Omega-omega2)/omega2)
            
        f = lambda Omega: Omega - np.sqrt( (mu*omega_r)**2 - alpha*e*omega_g*I/(E*T0) * mu*(omega_g*sigma_t)**(2*mu-2)/(2**mu*math.factorial(mu-1)) *(R1*F1**2*(np.sin(2*phi1_neg(Omega)) + np.sin(2*phi1_pos(Omega))) + nu**(2*mu-1)*R2*F2**2*(np.sin(2*phi2_neg(Omega)) + np.sin(2*phi2_pos(Omega))) ) )        
                                
        while np.absolute(f(Omega0)) > 1e-6:
                                    
            Omega1 = Omega0 - f(Omega0)/misc.derivative(f,Omega0,dx=1,n=1)
            Omega0 = Omega1
            
        Omega[mu-1] = Omega0
        alpha_r[mu-1] = 8*alpha*e*I/(E*T0)*mu*(omega_g*sigma_t)**(2*mu-2)/(2**mu*math.factorial(mu-1)) * (F1**2*R1*Q1*np.tan(phi1)*np.cos(phi1_pos(Omega0))**2*np.cos(phi1_neg(Omega0))**2 + nu**(2*mu-2)*F2**2*R2*Q2*np.tan(phi2)*np.cos(phi2_pos(Omega0))**2*np.cos(phi2_neg(Omega0))**2)            
            
#        # Initial guess based on zero current
#        Omega_guess = mu*omega_r
#        diff = 1
#                
#        while np.absolute(diff) > 1e-6:
#            
#            
#            phi1_pos = np.arctan(2*Q1*(omega_g+Omega_guess-omega1)/omega1)
#            phi1_neg = np.arctan(2*Q1*(omega_g-Omega_guess-omega1)/omega1)
#                                    
#            phi2_pos = np.arctan(2*Q2*(nu*omega_g+Omega_guess-omega2)/omega2)
#            phi2_neg = np.arctan(2*Q2*(nu*omega_g-Omega_guess-omega2)/omega2)
#            
#            Omega_temp = np.sqrt( (mu*omega_r)**2 - alpha*e*omega_g*I/(E*T0) * mu*(omega_g*sigma_t)**(2*mu-2)/(2**mu*math.factorial(mu-1)) *(R1*F1**2*(np.sin(2*phi1_neg) + np.sin(2*phi1_pos)) + nu**(2*mu-1)*R2*F2**2*(np.sin(2*phi2_neg) + np.sin(2*phi2_pos)) ) )
#            
#            diff = Omega_temp - Omega_guess
#            Omega_guess = Omega_guess + 0.1*diff
            
#        Omega[mu-1] = Omega_temp                    
#        alpha_r[mu-1] = 8*alpha*e*I/(E*T0)*mu*(omega_g*sigma_t)**(2*mu-2)/(2**mu*math.factorial(mu-1)) * (F1**2*R1*Q1*np.tan(phi1)*np.cos(phi1_pos)**2*np.cos(phi1_neg)**2 + nu**(2*mu-2)*F2**2*R2*Q2*np.tan(phi2)*np.cos(phi2_pos)**2*np.cos(phi2_neg)**2)
        
        # Add radiation damping to damping rates
        alpha_r_incl_rad_damp[mu-1] = alpha_r[mu-1] + mu/tauL
        
        # Complex frequency shift        
        delta_Omega[mu-1] = Omega[mu-1] - 1j*alpha_r[mu-1] - mu*omega_r
     
    # Check if dipole robinson instability exists   
    if alpha_r_incl_rad_damp[0] < 0:
        
        # Check if Landau damping is overcome
        if np.abs(delta_Omega[0]) > landau_threshold[0]:
            dipole_robinson = 1
        else:
            dipole_robinson = 0
            
    else:
        dipole_robinson = 0
        
    # Check if quadrupole robinson instability exists   
    if alpha_r_incl_rad_damp[1] < 0:
        
        # Check if Landau damping is overcome
        if np.abs(delta_Omega[1]) > landau_threshold[1]:
            quadrupole_robinson = 1
        else:
            quadrupole_robinson = 0
            
    else:
        quadrupole_robinson = 0

    # Check if sextupole robinson instability exists   
    if alpha_r_incl_rad_damp[2] < 0:
        
        # Check if Landau damping is overcome
        if np.abs(delta_Omega[2]) > landau_threshold[2]:
            sextupole_robinson = 1
        else:
            sextupole_robinson = 0
            
    else:
        sextupole_robinson = 0 

    # Check if octupole robinson instability exists   
    if alpha_r_incl_rad_damp[3] < 0:
        
        # Check if Landau damping is overcome
        if np.abs(delta_Omega[3]) > landau_threshold[3]:
            octupole_robinson = 1
        else:
            octupole_robinson = 0
            
    else:
        octupole_robinson = 0             
                                
    return zero_frequency,dipole_robinson,quadrupole_robinson,sextupole_robinson,octupole_robinson

def robinson_coupling(E,T0,alpha,sigma_e,frf,VT1,psi1,xi,nu,sigma_t,F1,F2,R1,Q1,phi1,R2,Q2,phi2,I,b,c,tauL):
    
    # Initilise return parameters
    zero_frequency_instability = np.nan
    fast_mode_coupling = np.nan
    coupled_dipole =  np.nan
    coupled_quadrupole =  np.nan
    
    omega_g = 2*np.pi*frf
    E = E*e # Change to SI units
    
    omega1 = 2*Q1*omega_g/(2*Q1+np.tan(phi1))
    omega2 = 2*Q2*nu*omega_g/(2*Q2+np.tan(phi2))
    
    # Robinson frequency
    omega_r = np.sqrt(alpha*e*omega_g/(E*T0)*(F1-xi*F2)*VT1*np.sin(psi1))
    
    phi1_pos = lambda Omega: np.arctan(2*Q1*(omega_g+Omega-omega1)/omega1)
    phi1_neg = lambda Omega: np.arctan(2*Q1*(omega_g-Omega-omega1)/omega1)
                                    
    phi2_pos = lambda Omega: np.arctan(2*Q2*(nu*omega_g+Omega-omega2)/omega2)
    phi2_neg = lambda Omega: np.arctan(2*Q2*(nu*omega_g-Omega-omega2)/omega2)
                                                                          
    A_tilde = lambda Omega: alpha*e*omega_g*I/(2*E*T0) * (R1*F1**2*(np.sin(2*phi1_neg(Omega))+np.sin(2*phi1_pos(Omega))) + nu*R2*F2**2*(np.sin(2*phi2_neg(Omega))+np.sin(2*phi2_pos(Omega))) )
        
    B_tilde = lambda Omega: alpha*e*omega_g*I/(2*E*T0)*(omega_g*sigma_t)**2 * (R1*F1**2*(np.sin(2*phi1_neg(Omega))+np.sin(2*phi1_pos(Omega))) + nu**3*R2*F2**2*(np.sin(2*phi2_neg(Omega))+np.sin(2*phi2_pos(Omega))) )
        
    D_tilde = lambda Omega: (omega_g*sigma_t)*alpha*e*omega_g*I/(2*E*T0) * (R1*F1**2*(np.sin(2*phi1_neg(Omega))-np.sin(2*phi1_pos(Omega))) + nu**2*R2*F2**2*(np.sin(2*phi2_neg(Omega))-np.sin(2*phi2_pos(Omega))) )
        
    a_tilde = lambda Omega: alpha*e*omega_g*I/(E*T0) * (R1*F1**2*(np.cos(phi1_neg(Omega))**2-np.cos(phi1_pos(Omega))**2) + nu*R2*F2**2*(np.cos(phi2_neg(Omega))**2-np.cos(phi2_pos(Omega))**2) ) + 2*Omega/tauL
        
    b_tilde = lambda Omega: alpha*e*omega_g*I/(E*T0)*(omega_g*sigma_t)**2 * (R1*F1**2*(np.cos(phi1_neg(Omega))**2-np.cos(phi1_pos(Omega))**2) + nu**3*R2*F2**2*(np.cos(phi2_neg(Omega))**2-np.cos(phi2_pos(Omega))**2) ) + 4*Omega/tauL
        
    d_tilde = lambda Omega: (omega_g*sigma_t)*alpha*e*omega_g*I/(E*T0) * (R1*F1**2*(np.cos(phi1_neg(Omega))**2+np.cos(phi1_pos(Omega))**2) + nu**2*R2*F2**2*(np.cos(phi2_neg(Omega))**2+np.cos(phi2_pos(Omega))**2) )
    
    # Zero frequency instability
    # These conditions are the same for no coupling and with coupling (comparison with A_tilde and B_tilde results in same formula) except for addition of coupled zero-frequency condition  
    
    if omega_r**2 - A_tilde(0) < 0:
        zero_frequency_instability = 1        
    elif 4*omega_r**2 - B_tilde(0) < 0:
        zero_frequency_instability = 2
    elif (omega_r**2 - A_tilde(0))*(4*omega_r**2 - B_tilde(0)) + d_tilde(0) < 0:
         zero_frequency_instability = 3    
    else:
        zero_frequency_instability = 0 
        
#    zero_frequency_instability_formula = lambda mu: alpha*e*omega_g*I/(E*T0)*(omega_g*sigma_t)**(2*mu-2)/(2**(mu-1)*math.factorial(mu))*(R1*F1**2*np.sin(2*phi1) + nu**(2*mu-1)*R2*F2**2*np.sin(2*phi2))
#    coupled_zero_frequency_instability_formula
#           
#    if omega_r**2 < zero_frequency_instability_formula(1):
#        zero_frequency_instability = 1        
#    elif omega_r**2 < zero_frequency_instability_formula(2):
#        zero_frequency_instability = 2
#    else:
#        zero_frequency_instability = 0
        
    # If zero frequency instability exists return without calculating AC instability        
    if zero_frequency_instability != 0:        
        return zero_frequency_instability,fast_mode_coupling,coupled_dipole,coupled_quadrupole  
                       
    # --- Calculate coupled-dipole mode  ---   
    Omega = np.zeros((2,1))
    alpha_r_incl_rad_damp = np.zeros((2,1))
    delta_Omega = np.zeros((2,1),dtype=complex)
                        
    # Solve using Newton-Raphson method
                            
    
#    alpha_r = lambda Omega: (a_tilde*(Omega**2-(2*omega_r)**2+B_tilde) + b_tilde*(Omega**2-omega_r**2+A_tilde) - 2*D_tilde*d_tilde)/(2*Omega*(2*Omega**2-5*omega_r**2+A_tilde+B_tilde))           
#    Omega_square = lambda Omega: (5*omega_r**2-A_tilde-B_tilde)/2 - np.sqrt( np.absolute((3*omega_r**2+A_tilde-B_tilde)**2/4 + D_tilde**2 - d_tilde**2 + (a_tilde-2*Omega*alpha_r)*(b_tilde-2*Omega*alpha_r)))
    
    f1 = lambda Omega,alpha_r: alpha_r - ( (a_tilde(Omega)*(Omega**2-(2*omega_r)**2+B_tilde(Omega)) + b_tilde(Omega)*(Omega**2-omega_r**2+A_tilde(Omega)) - 2*D_tilde(Omega)*d_tilde(Omega))/(2*Omega*(2*Omega**2-5*omega_r**2+A_tilde(Omega)+B_tilde(Omega))) )
    f2 = lambda Omega,alpha_r: Omega - np.sqrt( (5*omega_r**2-A_tilde(Omega)-B_tilde(Omega))/2 - np.sqrt( np.absolute((3*omega_r**2+A_tilde(Omega)-B_tilde(Omega))**2/4 + D_tilde(Omega)**2 - d_tilde(Omega)**2 + (a_tilde(Omega)-2*Omega*alpha_r)*(b_tilde(Omega)-2*Omega*alpha_r))) )

#    f2 = lambda Omega,alpha_r: Omega - np.sqrt( (5*omega_r**2-A_tilde(Omega)-B_tilde(Omega))/2 - np.sqrt( ((3*omega_r**2+A_tilde(Omega)-B_tilde(Omega))**2/4 + D_tilde(Omega)**2 - d_tilde(Omega)**2 + (a_tilde(Omega)-2*Omega*alpha_r)*(b_tilde(Omega)-2*Omega*alpha_r))) )
    
    def f(f1,f2,point):
        
        x1 = point[0].item()
        x2 = point[1].item()
                
        result = np.array([[f1(x1,x2)],[f2(x1,x2)]])                   
        return result
    
    def jacobian(f1,f2,point):
        
        x1 = point[0].item()
        x2 = point[1].item()
                
        def partial_derivative(func, var=0, point=[]):
            args = point[:]
            def wraps(x):
                args[var] = x
                return func(*args)
            return misc.derivative(wraps, point[var], dx = 1e-6)
        
        result = np.array([ [partial_derivative(f1, 0, [x1,x2]),partial_derivative(f1, 1, [x1,x2])],[partial_derivative(f2, 0, [x1,x2]),partial_derivative(f2, 1, [x1,x2])] ])
        return result
    
    # Initial guess based on zero current
    alpha_r0 = 1/tauL
    Omega0 = omega_r
    x0 = np.array([[Omega0],[alpha_r0]])
               
    while np.any(np.absolute(f(f1,f2,x0)) > 1e-6):
        
#        # Check fast-mode coupling instability         
#        fast_mode_coupling_formula = lambda Omega,alpha_r: (3*omega_r**2+A_tilde(Omega)-B_tilde(Omega))**2/4 + D_tilde(Omega)**2 - d_tilde(Omega)**2 + (a_tilde(Omega)-2*Omega*alpha_r)*(b_tilde(Omega)-2*Omega*alpha_r)
##        fast_mode_coupling_formula = lambda Omega,alpha_r: (5*omega_r**2-A_tilde(Omega)-B_tilde(Omega))(3*omega_r**2+A_tilde(Omega)-B_tilde(Omega))**2/4 + D_tilde(Omega)**2 - d_tilde(Omega)**2 + (a_tilde(Omega)-2*Omega*alpha_r)*(b_tilde(Omega)-2*Omega*alpha_r)
#    
#        if fast_mode_coupling_formula(x0[0],x0[1]) < 0:
#            fast_mode_coupling = 1
#            return zero_frequency_instability,fast_mode_coupling,coupled_dipole,coupled_quadrupole
#        else:
#            fast_mode_coupling = 0
                
        # Calculate the Jacobian
        J = jacobian(f1,f2,x0)
                        
        x1 = x0 -  np.matmul(np.linalg.inv(J),f(f1,f2,x0))          
        x0 = x1
        
    Omega[0] = x0[0]
         
    alpha_r_incl_rad_damp[0] = x0[1]   
    delta_Omega[0] = Omega[0] - 1j*alpha_r_incl_rad_damp[0] - omega_r
            
    # Check fast mode-coupling instability    
    fast_mode_coupling_formula = lambda Omega,alpha_r: (3*omega_r**2+A_tilde(Omega)-B_tilde(Omega))**2/4 + D_tilde(Omega)**2 - d_tilde(Omega)**2 + (a_tilde(Omega)-2*Omega*alpha_r)*(b_tilde(Omega)-2*Omega*alpha_r)
    
    if fast_mode_coupling_formula(Omega[0],alpha_r_incl_rad_damp[0]) < 0:
        fast_mode_coupling = 1
    else:
        fast_mode_coupling = 0
                
 
        
    
#        Omega[mu-1] = Omega0
#        alpha_r[mu-1] = 8*alpha*e*I/(E*T0)*mu*(omega_g*sigma_t)**(2*mu-2)/(2**mu*math.factorial(mu-1)) * (F1**2*R1*Q1*np.tan(phi1)*np.cos(phi1_pos(Omega0))**2*np.cos(phi1_neg(Omega0))**2 + nu**(2*mu-2)*F2**2*R2*Q2*np.tan(phi2)*np.cos(phi2_pos(Omega0))**2*np.cos(phi2_neg(Omega0))**2)            
        
        #            Omega1 = Omega0 - f(Omega0)/misc.derivative(f,Omega0,dx=1,n=1)
#            Omega0 = Omega1
    
#    f = lambda Omega: 
    
    
    
    
    
        
#        phi1_pos = lambda Omega: np.arctan(2*Q1*(omega_g+Omega-omega1)/omega1)
#        phi1_neg = lambda Omega: np.arctan(2*Q1*(omega_g-Omega-omega1)/omega1)
#                                    
#        phi2_pos = lambda Omega: np.arctan(2*Q2*(nu*omega_g+Omega-omega2)/omega2)
#        phi2_neg = lambda Omega: np.arctan(2*Q2*(nu*omega_g-Omega-omega2)/omega2)
#            
#        f = lambda Omega: Omega - np.sqrt( (mu*omega_r)**2 - alpha*e*omega_g*I/(E*T0) * mu*(omega_g*sigma_t)**(2*mu-2)/(2**mu*math.factorial(mu-1)) *(R1*F1**2*(np.sin(2*phi1_neg(Omega)) + np.sin(2*phi1_pos(Omega))) + nu**(2*mu-1)*R2*F2**2*(np.sin(2*phi2_neg(Omega)) + np.sin(2*phi2_pos(Omega))) ) )        
#                                
#        while np.absolute(f(Omega0)) > 1e-6:
#                                    
#            Omega1 = Omega0 - f(Omega0)/misc.derivative(f,Omega0,dx=1,n=1)
#            Omega0 = Omega1
    
    
    
    
    
    
    
    
    
#    Omega_guess = omega_r
#    alpha_r_guess = 1/tauL
#    diff = 1
    
#    while np.absolute(diff) > 1e-6:
#        
#        phi1_pos = np.arctan(2*Q1*(omega_g+Omega_guess-omega1)/omega1)
#        phi1_neg = np.arctan(2*Q1*(omega_g-Omega_guess-omega1)/omega1)
#                                    
#        phi2_pos = np.arctan(2*Q2*(nu*omega_g+Omega_guess-omega2)/omega2)
#        phi2_neg = np.arctan(2*Q2*(nu*omega_g-Omega_guess-omega2)/omega2)
#                                                                          
#        A_tilde = alpha*e*omega_g*I/(2*E*T0) * (R1*F1**2*(np.sin(2*phi1_neg)+np.sin(2*phi1_pos)) + nu*R2*F2**2*(np.sin(2*phi2_neg)+np.sin(2*phi2_pos)) )
#        
#        B_tilde = alpha*e*omega_g*I/(2*E*T0)*(omega_g*sigma_t)**2 * (R1*F1**2*(np.sin(2*phi1_neg)+np.sin(2*phi1_pos)) + nu**3*R2*F2**2*(np.sin(2*phi2_neg)+np.sin(2*phi2_pos)) )
#        
#        D_tilde = (omega_g*sigma_t)*alpha*e*omega_g*I/(2*E*T0) * (R1*F1**2*(np.sin(2*phi1_neg)-np.sin(2*phi1_pos)) + nu**2*R2*F2**2*(np.sin(2*phi2_neg)-np.sin(2*phi2_pos)) )
#        
#        a_tilde = alpha*e*omega_g*I/(E*T0) * (R1*F1**2*(np.cos(phi1_neg)**2-np.cos(phi1_pos)**2) + nu*R2*F2**2*(np.cos(phi2_neg)**2-np.cos(phi2_pos)**2) ) + 2*Omega_guess/tauL
#        
#        b_tilde = alpha*e*omega_g*I/(E*T0)*(omega_g*sigma_t)**2 * (R1*F1**2*(np.cos(phi1_neg)**2-np.cos(phi1_pos)**2) + nu**3*R2*F2**2*(np.cos(phi2_neg)**2-np.cos(phi2_pos)**2) ) + 4*Omega_guess/tauL
#        
#        d_tilde = (omega_g*sigma_t)*alpha*e*omega_g*I/(E*T0) * (R1*F1**2*(np.cos(phi1_neg)**2+np.cos(phi1_pos)**2) + nu**2*R2*F2**2*(np.cos(phi2_neg)**2+np.cos(phi2_pos)**2) )
#                    
#        alpha_r = (a_tilde*(Omega_guess**2-(2*omega_r)**2+B_tilde) + b_tilde*(Omega_guess**2-omega_r**2+A_tilde) - 2*D_tilde*d_tilde)/(2*Omega_guess*(2*Omega_guess**2-5*omega_r**2+A_tilde+B_tilde))   
#        
#        Omega_square = (5*omega_r**2-A_tilde-B_tilde)/2 - np.sqrt( np.absolute((3*omega_r**2+A_tilde-B_tilde)**2/4 + D_tilde**2 - d_tilde**2 + (a_tilde-2*Omega_guess*alpha_r)*(b_tilde-2*Omega_guess*alpha_r)))
#        
#        #Omega_square = (5*omega_r**2-A_tilde-B_tilde)/2 - np.sqrt( np.absolute((3*omega_r**2+A_tilde-B_tilde)**2/4 + D_tilde**2 - d_tilde**2 + (a_tilde-2*Omega_guess*alpha_r_guess)*(b_tilde-2*Omega_guess*alpha_r_guess)))
#        diff = np.sqrt(Omega_square) - Omega_guess
#        Omega_guess = Omega_guess + 0.1*diff
        
    
    # Check fast mode-coupling instability    
#    if (3*omega_r**2+A_tilde-B_tilde)**2/4 + D_tilde**2 - d_tilde**2 + (a_tilde-2*Omega_guess*alpha_r)*(b_tilde-2*Omega_guess*alpha_r) < 0:
#        fast_mode_coupling = 1
#    else:
#        fast_mode_coupling = 0
#                
#    Omega[0] = Omega_guess
#    alpha_r_incl_rad_damp[0] = alpha_r  
#    delta_Omega[0] = Omega[0] - 1j*alpha_r_incl_rad_damp[0] - omega_r
    
#    # Calculate coupled-quadrupole mode      
#    Omega_guess = 2*omega_r
##    alpha_r_guess = 1/tauL
#    diff = 1
#    
#    while np.absolute(diff) > 1e-6:
#        
#        phi1_pos = np.arctan(2*Q1*(omega_g+Omega_guess-omega1)/omega1)
#        phi1_neg = np.arctan(2*Q1*(omega_g-Omega_guess-omega1)/omega1)
#                                    
#        phi2_pos = np.arctan(2*Q2*(nu*omega_g+Omega_guess-omega2)/omega2)
#        phi2_neg = np.arctan(2*Q2*(nu*omega_g-Omega_guess-omega2)/omega2)
#                                                                          
#        A_tilde = alpha*e*omega_g*I/(2*E*T0) * (R1*F1**2*(np.sin(2*phi1_neg)+np.sin(2*phi1_pos)) + nu*R2*F2**2*(np.sin(2*phi2_neg)+np.sin(2*phi2_pos)) )
#        
#        B_tilde = alpha*e*omega_g*I/(2*E*T0)*(omega_g*sigma_t)**2 * (R1*F1**2*(np.sin(2*phi1_neg)+np.sin(2*phi1_pos)) + nu**3*R2*F2**2*(np.sin(2*phi2_neg)+np.sin(2*phi2_pos)) )
#        
#        D_tilde = (omega_g*sigma_t)*alpha*e*omega_g*I/(2*E*T0) * (R1*F1**2*(np.sin(2*phi1_neg)-np.sin(2*phi1_pos)) + nu**2*R2*F2**2*(np.sin(2*phi2_neg)-np.sin(2*phi2_pos)) )
#        
#        a_tilde = alpha*e*omega_g*I/(E*T0) * (R1*F1**2*(np.cos(phi1_neg)**2-np.cos(phi1_pos)**2) + nu*R2*F2**2*(np.cos(phi2_neg)**2-np.cos(phi2_pos)**2) ) + 2*Omega_guess/tauL
#        
#        b_tilde = alpha*e*omega_g*I/(E*T0)*(omega_g*sigma_t)**2 * (R1*F1**2*(np.cos(phi1_neg)**2-np.cos(phi1_pos)**2) + nu**3*R2*F2**2*(np.cos(phi2_neg)**2-np.cos(phi2_pos)**2) ) + 4*Omega_guess/tauL
#        
#        d_tilde = (omega_g*sigma_t)*alpha*e*omega_g*I/(E*T0) * (R1*F1**2*(np.cos(phi1_neg)**2+np.cos(phi1_pos)**2) + nu**2*R2*F2**2*(np.cos(phi2_neg)**2+np.cos(phi2_pos)**2) )
#            
#        alpha_r = (a_tilde*(Omega_guess**2-(2*omega_r)**2+B_tilde) + b_tilde*(Omega_guess**2-omega_r**2+A_tilde) - 2*D_tilde*d_tilde)/(2*Omega_guess*(2*Omega_guess**2-5*omega_r**2+A_tilde+B_tilde))   
#        
#        Omega_square = (5*omega_r**2-A_tilde-B_tilde)/2 + np.sqrt( np.absolute((3*omega_r**2+A_tilde-B_tilde)**2/4 + D_tilde**2 - d_tilde**2 + (a_tilde-2*Omega_guess*alpha_r)*(b_tilde-2*Omega_guess*alpha_r)))
#        
#        #Omega_square = (5*omega_r**2-A_tilde-B_tilde)/2 - np.sqrt( np.absolute((3*omega_r**2+A_tilde-B_tilde)**2/4 + D_tilde**2 - d_tilde**2 + (a_tilde-2*Omega_guess*alpha_r_guess)*(b_tilde-2*Omega_guess*alpha_r_guess)))
#        diff = np.sqrt(Omega_square) - Omega_guess
#        Omega_guess = Omega_guess + 0.1*diff
#        
#    Omega[1] = Omega_guess
#    alpha_r_incl_rad_damp[1] = alpha_r  
#    delta_Omega[1] = Omega[1] - 1j*alpha_r_incl_rad_damp[1] - 2*omega_r    
    
    
    # Landau threshold
    dipole_landau = 0.78*alpha**2*sigma_e**2/omega_r*np.absolute(3*c/omega_r**2 - (2*b/omega_r**2)**2)
        
    landau_threshold = np.zeros((2,1))
    landau_threshold[0] = dipole_landau
    landau_threshold[1] = 2.24/0.78*dipole_landau
    
    # Check if coupled-dipole instability exists   
    if alpha_r_incl_rad_damp[0] < 0:
        
        # Check if Landau damping is overcome
        if np.abs(delta_Omega[0]) > landau_threshold[0]:
            coupled_dipole = 1
        else:
            coupled_dipole = 0
            
    else:
        coupled_dipole = 0
        
#    # Check if coupled-quadrupole instability exists   
#    if alpha_r_incl_rad_damp[1] < 0:
#        
#        # Check if Landau damping is overcome
#        if np.abs(delta_Omega[1]) > landau_threshold[1]:
#            coupled_quadrupole = 1
#        else:
#            coupled_quadrupole = 0
#            
#    else:
#        coupled_quadrupole = 0
        
    return zero_frequency_instability,fast_mode_coupling,coupled_dipole,coupled_quadrupole

def dipole_coupled_bunch(E,T0,alpha,sigma_e,frf,VT1,psi1,xi,nu,sigma_t,F1,F2,R1,Q1,phi1,R2,Q2,phi2,I,b,c,tauL):
    
    omega_g = 2*np.pi*frf
    E = E*e # Change to SI units
    
    omega1 = 2*Q1*omega_g/(2*Q1+np.tan(phi1))
    omega2 = 2*Q2*nu*omega_g/(2*Q2+np.tan(phi2))
    
    # Robinson frequency
    omega_r = np.sqrt(alpha*e*omega_g/(E*T0)*(F1-xi*F2)*VT1*np.sin(psi1))
    
    CB_modes = np.array([-1,1])
    
    Omega_CB = np.zeros((2,1))
    alpha_r_CB = np.zeros((2,1))
    alpha_r_CB_incl_rad_damp = np.zeros((2,1))
    delta_Omega_CB = np.zeros((2,1),dtype=complex)
    
    omega0 = 2*np.pi/T0
    
    mu = 1
    
    for i in range(len(CB_modes)):
                       
        # Initial guess
        Omega_guess = mu*omega_r
        diff = 1
        
        while np.absolute(diff) > 1e-6:
            
            phi1_pos = np.arctan(2*Q1*(omega_g+(Omega_guess+CB_modes[i]*omega0)-omega1)/omega1)
            phi1_neg = np.arctan(2*Q1*(omega_g-(Omega_guess+CB_modes[i]*omega0)-omega1)/omega1)

            phi2_pos = np.arctan(2*Q2*(nu*omega_g+(Omega_guess+CB_modes[i]*omega0)-omega2)/omega2)
            phi2_neg = np.arctan(2*Q2*(nu*omega_g-(Omega_guess+CB_modes[i]*omega0)-omega2)/omega2)
            
            Omega_temp = np.sqrt( (mu*omega_r)**2 - alpha*e*omega_g*I/(E*T0) * mu*(omega_g*sigma_t)**(2*mu-2)/(2**mu*math.factorial(mu-1)) *(R1*F1**2*(np.sin(2*phi1_neg) + np.sin(2*phi1_pos)) + nu**(2*mu-1)*R2*F2**2*(np.sin(2*phi2_neg) + np.sin(2*phi2_pos)) ) )
            
            diff = Omega_temp - Omega_guess
            Omega_guess = Omega_guess + 0.1*diff
                    
        Omega_CB[i] = Omega_temp
                
        alpha_r_CB[i] = alpha*e*omega_g*I/(Omega_CB[i]*E*T0)*mu*(omega_g*sigma_t)**(2.*mu-2)/(2**mu*math.factorial(mu-1)) * (F1**2*R1*(np.cos(phi1_neg)**2 - np.cos(phi1_pos)**2) + nu**(2*mu-1)*F2**2*R2*(np.cos(phi2_neg)**2 - np.cos(phi2_pos)**2) )   
                
        # Add radiation damping to damping rates
        alpha_r_CB_incl_rad_damp[i] = alpha_r_CB[i] + mu/tauL
               
        # Complex frequency shift        
        #delta_Omega_CB[i] = (Omega_CB[i]+CB_modes[i]*omega0) - 1j*alpha_r_CB[i] - mu*omega_r  
        delta_Omega_CB[i] = (Omega_CB[i]) - 1j*alpha_r_CB[i] - mu*omega_r
    

    # Landau threshold
    dipole_landau = 0.78*alpha**2*sigma_e**2/omega_r*np.absolute(3*c/omega_r**2 - (2*b/omega_r**2)**2)         
            
    # Check dipole instability    
    if alpha_r_CB_incl_rad_damp[0] < 0:
        
        #Check if Landau damping is overcome 
        if np.absolute(delta_Omega_CB[0]) > dipole_landau:
            CB_neg_dipole_instability_exist = 1
        else:
            CB_neg_dipole_instability_exist = 0
    else:
        CB_neg_dipole_instability_exist = 0
        
    if alpha_r_CB_incl_rad_damp[1] < 0:
        
        #Check if Landau damping is overcome 
        if np.absolute(delta_Omega_CB[1]) > dipole_landau:
            CB_pos_dipole_instability_exist = 1
        else:
            CB_pos_dipole_instability_exist = 0
    else:
        CB_pos_dipole_instability_exist = 0
        
    if CB_neg_dipole_instability_exist or CB_pos_dipole_instability_exist:
        CB_dipole_instability_exist = 1
    else:
        CB_dipole_instability_exist = 0
        
    return CB_dipole_instability_exist
       