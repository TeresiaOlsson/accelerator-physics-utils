#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 10:13:10 2021

@author: Teresia Olsson, teresia.olsson@helmholtz-berlin.de
"""

import numpy as np
from scipy.constants import elementary_charge as e
import scipy.integrate as integrate
import math
import scipy.misc as misc

class MachineSettings:
    
    def __init__(self, energy:float, rev_time:float, mom_comp:float, energy_loss:float, energy_spread:float, long_damping:float):
        
        self.energy = energy
        self.rev_time = rev_time
        self.mom_comp = mom_comp
        self.energy_loss = energy_loss
        self.energy_spread = energy_spread
        self.long_damping = long_damping
        
class CavitySettings:
    
    def __init__(self, rf_freq:float, V_main:float, R_main:float, Q_main:float, beta_main:float, n_harm:float, R_harm:float, Q_harm:float, beta_harm:float):
        
        self.rf_freq = rf_freq
        self.V_main = V_main
        self.R_main = R_main
        self.Q_main = Q_main
        self.beta_main = beta_main
        self.n_harm = n_harm
        self.R_harm = R_harm
        self.Q_harm = Q_harm
        self.beta_harm = beta_harm
               
def phases_symmetric_profile(VT1: float, nu:float, xi:float, Vs:float) -> (float,float):
    """
    
    Calculate synchronous phase angles and harmonic cavity voltage based on assuming cubic term to be zero = approximately symmetric bunch

    Args:
        VT1 (float): Main cavity voltage.
        nu (float): Harmonic cavity harmonic.
        xi (float): Negative of ratio between harmonic and main cavity longitudinal force: -nHarm*V_HC*sin(psi_HC) / ( V_MC*sin(psi_MC) ) .
        Vs (float): Energy loss per turn.

    Returns:
        (float,float): Harmonic cavity phase (psi_HC), Main cavity phase (psi_MC).

    """       
    psi1 = np.arccos(Vs/(1-1/nu**2)/VT1)    
    psi2 = np.arctan(nu*xi*np.tan(psi1)) - np.pi # Why - pi?
    
    return (psi1,psi2)

def equilibrium_phase_exists(psi1:float) -> bool:
    """
    
    Args:
        psi1 (float): Main cavity phase.

    Returns:
        bool: 0: unstable phase, 1: stable phase.

    """     
    # If synchronous phase not between 0 and 90 degress no phase focusing exists
    if psi1 < 0 or psi1 > np.pi/2:
        return False
    else:
        return True
    
def potential_coefficients(E,T0,alpha,frf,VT1,psi1,nu,VT2,psi2):
    
    omega_g = 2*np.pi*frf
    E = E*e # Change to SI units
    
    a = alpha*e*omega_g/(2*E*T0)*(VT1*np.sin(psi1)+nu*VT2*np.sin(psi2))
    b = alpha*e*omega_g**2/(6*E*T0)*(VT1*np.cos(psi1)+nu**2*VT2*np.cos(psi2))
    c = -alpha*e*omega_g**3/(24*E*T0)*(VT1*np.sin(psi1)+nu**3*VT2*np.sin(psi2))
           
    return a,b,c

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

def estimated_robinson_freq(machine,cavities,xi,formfactor_main,formfactor_harm,phase_main):
    
    # Change to notation used in paper in Sec II. B & C
    alpha = machine.mom_comp
    frf = cavities.rf_freq
    omega_g = 2*np.pi*frf
    E = machine.energy*e # Change to SI units
    T0 = machine.rev_time
    F1 = formfactor_main
    F2 = formfactor_harm
    VT1 = cavities.V_main
    psi1 = phase_main
    
    return np.sqrt(alpha*e*omega_g/(E*T0)*(F1-xi*F2)*VT1*np.sin(psi1))

def zero_frequency_instability_no_coupling(machine,cavities,current,bunch_length,R_main_loaded,formfactor_main,tuning_angle_main,R_harm_loaded,formfactor_harm,tuning_angle_harm,omega_r):
    # Check of zero frequency instability is predicted from modes 1-4
    #(Eq. 14)
    
    # Change to notation used in paper in Sec II. B & C
    alpha = machine.mom_comp
    frf = cavities.rf_freq
    omega_g = 2*np.pi*frf
    I = current
    E = machine.energy*e # Change to SI units
    T0 = machine.rev_time
    sigma_t = bunch_length
    R1 = R_main_loaded
    F1 = formfactor_main
    phi1 = tuning_angle_main
    nu = cavities.n_harm
    R2 = R_harm_loaded
    F2 = formfactor_harm   
    phi2 = tuning_angle_harm
   
    zero_frequency = lambda mu: alpha*e*omega_g*I/(E*T0)*(omega_g*sigma_t)**(2*mu-2)/(2**(mu-1)*math.factorial(mu))*(R1*F1**2*np.sin(2*phi1) + nu**(2*mu-1)*R2*F2**2*np.sin(2*phi2))
           
    if omega_r**2 < zero_frequency(1):
        return 1
    elif omega_r**2 < zero_frequency(2):
        return 2
    elif omega_r**2 < zero_frequency(3):        
        return 3        
    elif omega_r**2 < zero_frequency(4):
        return 4
    else:
        return False
        
        
def zero_frequency_instability_coupling(machine,cavities,current,bunch_length,R_main_loaded,Q_main_loaded,formfactor_main,tuning_angle_main,R_harm_loaded,Q_harm_loaded,formfactor_harm,tuning_angle_harm,omega_r):
    # Appendix B, Eq. B8
    
    # Change to notation used in paper in Sec II. B & C
    alpha = machine.mom_comp
    frf = cavities.rf_freq
    omega_g = 2*np.pi*frf
    I = current
    E = machine.energy*e # Change to SI units
    T0 = machine.rev_time
    sigma_t = bunch_length
    R1 = R_main_loaded
    Q1 = Q_main_loaded
    F1 = formfactor_main
    phi1 = tuning_angle_main
    nu = cavities.n_harm
    R2 = R_harm_loaded
    Q2 = Q_harm_loaded
    F2 = formfactor_harm   
    phi2 = tuning_angle_harm
       
    omega1 = 2*Q1*omega_g/(2*Q1+np.tan(phi1))
    omega2 = 2*Q2*nu*omega_g/(2*Q2+np.tan(phi2))
        
    phi1_pos = lambda Omega: np.arctan(2*Q1*(omega_g+Omega-omega1)/omega1)
    phi1_neg = lambda Omega: np.arctan(2*Q1*(omega_g-Omega-omega1)/omega1)
                                
    phi2_pos = lambda Omega: np.arctan(2*Q2*(nu*omega_g+Omega-omega2)/omega2)
    phi2_neg = lambda Omega: np.arctan(2*Q2*(nu*omega_g-Omega-omega2)/omega2)
                                                                      
    A_tilde = lambda Omega: alpha*e*omega_g*I/(2*E*T0) * (R1*F1**2*(np.sin(2*phi1_neg(Omega))+np.sin(2*phi1_pos(Omega))) + nu*R2*F2**2*(np.sin(2*phi2_neg(Omega))+np.sin(2*phi2_pos(Omega))) )
    
    B_tilde = lambda Omega: alpha*e*omega_g*I/(2*E*T0)*(omega_g*sigma_t)**2 * (R1*F1**2*(np.sin(2*phi1_neg(Omega))+np.sin(2*phi1_pos(Omega))) + nu**3*R2*F2**2*(np.sin(2*phi2_neg(Omega))+np.sin(2*phi2_pos(Omega))) )
      
    d_tilde = lambda Omega: (omega_g*sigma_t)*alpha*e*omega_g*I/(E*T0) * (R1*F1**2*(np.cos(phi1_neg(Omega))**2+np.cos(phi1_pos(Omega))**2) + nu**2*R2*F2**2*(np.cos(phi2_neg(Omega))**2+np.cos(phi2_pos(Omega))**2) )

    # These conditions are the same for no coupling and with coupling (comparison with A_tilde and B_tilde results in same formula) except for addition of coupled zero-frequency condition
    if omega_r**2 - A_tilde(0) < 0:
        return 1
    elif 4*omega_r**2 - B_tilde(0) < 0:
        return 2
    elif (omega_r**2 - A_tilde(0))*(4*omega_r**2 - B_tilde(0)) + d_tilde(0) < 0:       
        return 3        
    else:
        return False     
    
def landau_threshold(machine,omega_r,c,b):
     
    # Change to notation used in paper in Sec II. B & C
    alpha = machine.mom_comp
    sigma_e = machine.energy_spread
     
    # ---- Landau threshold ----
    dipole_landau = 0.78*alpha**2*sigma_e**2/omega_r*np.absolute(3*c/omega_r**2 - (2*b/omega_r**2)**2)
        
    landau_threshold = np.zeros((4,1))
    landau_threshold[0] = dipole_landau
    landau_threshold[1] = 2.24/0.78*dipole_landau
    landau_threshold[2] = 4.12/0.78*dipole_landau
    landau_threshold[3] = 6.36/0.78*dipole_landau
    
    return landau_threshold

def ac_robinson_no_coupling(machine,cavities,current,bunch_length,R_main_loaded,Q_main_loaded,formfactor_main,tuning_angle_main,R_harm_loaded,Q_harm_loaded,formfactor_harm,tuning_angle_harm,omega_r,landau_threshold):
    
    # Change to notation used in paper in Sec II. B & C
    frf = cavities.rf_freq
    omega_g = 2*np.pi*frf
    R1 = R_main_loaded
    F1 = formfactor_main
    Q1 = Q_main_loaded
    phi1 = tuning_angle_main
    nu = cavities.n_harm
    R2 = R_harm_loaded
    F2 = formfactor_harm 
    Q2 = Q_harm_loaded
    phi2 = tuning_angle_harm
    alpha = machine.mom_comp
    I = current
    E = machine.energy*e # Change to SI units
    T0 = machine.rev_time
    sigma_t = bunch_length
    tauL = machine.long_damping
    
    omega1 = 2*Q1*omega_g/(2*Q1+np.tan(phi1))
    omega2 = 2*Q2*nu*omega_g/(2*Q2+np.tan(phi2))
    
    # Calculate AC robinson growth rates    
    modes = np.array([1,2,3,4])
    
    # Output array
    ac_robinson = np.zeros(len(modes),dtype=bool)
    
    # Calculate the Robinson frequency
    for mu in modes:
                
        # Initial guess based on zero current
        Omega0 = mu*omega_r
        
        # Solve using Newton-Raphson method
        phi1_pos = lambda Omega: np.arctan(2*Q1*(omega_g+Omega-omega1)/omega1)
        phi1_neg = lambda Omega: np.arctan(2*Q1*(omega_g-Omega-omega1)/omega1)
                                    
        phi2_pos = lambda Omega: np.arctan(2*Q2*(nu*omega_g+Omega-omega2)/omega2)
        phi2_neg = lambda Omega: np.arctan(2*Q2*(nu*omega_g-Omega-omega2)/omega2)
        
        # Eq. 13
        f = lambda Omega: Omega - np.sqrt( (mu*omega_r)**2 - alpha*e*omega_g*I/(E*T0) * mu*(omega_g*sigma_t)**(2*mu-2)/(2**mu*math.factorial(mu-1)) *(R1*F1**2*(np.sin(2*phi1_neg(Omega)) + np.sin(2*phi1_pos(Omega))) + nu**(2*mu-1)*R2*F2**2*(np.sin(2*phi2_neg(Omega)) + np.sin(2*phi2_pos(Omega))) ) )        
                                
        while np.absolute(f(Omega0)) > 1e-6:                           
            Omega1 = Omega0 - f(Omega0)/misc.derivative(f,Omega0,dx=1,n=1)
            Omega0 = Omega1
        
        # Save the final result
        Omega = Omega0
            
        # When the Robinson frequency is found calculate the damping rate
        alpha_r = 8*alpha*e*I/(E*T0)*mu*(omega_g*sigma_t)**(2*mu-2)/(2**mu*math.factorial(mu-1)) * (F1**2*R1*Q1*np.tan(phi1)*np.cos(phi1_pos(Omega0))**2*np.cos(phi1_neg(Omega0))**2 + nu**(2*mu-2)*F2**2*R2*Q2*np.tan(phi2)*np.cos(phi2_pos(Omega0))**2*np.cos(phi2_neg(Omega0))**2)            
        
        # Add radiation damping to the damping rate
        alpha_r_incl_rad_damp= alpha_r + mu/tauL
        
        # Complex frequency shift        
        delta_Omega = Omega - 1j*alpha_r - mu*omega_r
        
        # Check if the instability occur
        if alpha_r_incl_rad_damp < 0:            
            # Check if Landau damping is overcome
            if np.abs(delta_Omega) > landau_threshold[mu-1]:
                ac_robinson[mu-1] = True
            else:
                ac_robinson[mu-1] = False  
        else:
            ac_robinson[mu-1] = False
        
    return ac_robinson

def ac_robinson_coupling(machine,cavities,current,bunch_length,R_main_loaded,Q_main_loaded,formfactor_main,tuning_angle_main,R_harm_loaded,Q_harm_loaded,formfactor_harm,tuning_angle_harm,omega_r,landau_threshold):
    
    # Change to notation used in paper in Sec II. B & C
    frf = cavities.rf_freq
    omega_g = 2*np.pi*frf
    R1 = R_main_loaded
    F1 = formfactor_main
    Q1 = Q_main_loaded
    phi1 = tuning_angle_main
    nu = cavities.n_harm
    R2 = R_harm_loaded
    F2 = formfactor_harm 
    Q2 = Q_harm_loaded
    phi2 = tuning_angle_harm
    alpha = machine.mom_comp
    I = current
    E = machine.energy*e # Change to SI units
    T0 = machine.rev_time
    sigma_t = bunch_length
    tauL = machine.long_damping
    
    omega1 = 2*Q1*omega_g/(2*Q1+np.tan(phi1))
    omega2 = 2*Q2*nu*omega_g/(2*Q2+np.tan(phi2))
        
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
    
  
    f1 = lambda Omega,alpha_r: alpha_r - ( (a_tilde(Omega)*(Omega**2-(2*omega_r)**2+B_tilde(Omega)) + b_tilde(Omega)*(Omega**2-omega_r**2+A_tilde(Omega)) - 2*D_tilde(Omega)*d_tilde(Omega))/(2*Omega*(2*Omega**2-5*omega_r**2+A_tilde(Omega)+B_tilde(Omega))) )
    f2 = lambda Omega,alpha_r: Omega - np.sqrt( (5*omega_r**2-A_tilde(Omega)-B_tilde(Omega))/2 - np.sqrt( np.absolute((3*omega_r**2+A_tilde(Omega)-B_tilde(Omega))**2/4 + D_tilde(Omega)**2 - d_tilde(Omega)**2 + (a_tilde(Omega)-2*Omega*alpha_r)*(b_tilde(Omega)-2*Omega*alpha_r))) )
    # An absolute value has been added in this formula to make the calculation stable
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
    x0 = np.array([[Omega0],[alpha_r0]]) #  Combine in one criteria
        
    while np.any(np.absolute(f(f1,f2,x0)) > 1e-6):
                        
        # Calculate the Jacobian
        J = jacobian(f1,f2,x0)
   
        # This has to be fixed but required at the moment to make the algorithm more stable
        try:                
            x1 = x0 -  np.matmul(np.linalg.inv(J),f(f1,f2,x0))          
            x0 = x1
        except:
            break

    # Save the final result
    Omega = x0[0]
    alpha_r_incl_rad_damp = x0[1]   
    delta_Omega = Omega - 1j*alpha_r_incl_rad_damp - omega_r

    # Check fast mode-coupling instability    
    fast_mode_coupling_formula = lambda Omega,alpha_r: (3*omega_r**2+A_tilde(Omega)-B_tilde(Omega))**2/4 + D_tilde(Omega)**2 - d_tilde(Omega)**2 + (a_tilde(Omega)-2*Omega*alpha_r)*(b_tilde(Omega)-2*Omega*alpha_r)
    
    if fast_mode_coupling_formula(Omega,alpha_r_incl_rad_damp) < 0:
        fast_mode_coupling = True
    else:
        fast_mode_coupling = False
        
    # Check if coupled-dipole instability exists   
    if alpha_r_incl_rad_damp < 0:
    
        # Check if Landau damping is overcome
        if np.abs(delta_Omega) > landau_threshold[0]:
            coupled_dipole = True
        else:
            coupled_dipole = False
        
    else:
        coupled_dipole = False
    
    # Coupled quadrupole not implemented yet
    coupled_quadrupole = None   
    
    return np.array([coupled_dipole, coupled_quadrupole, fast_mode_coupling])
          
def algorithm_active_cavity(machine: MachineSettings, cavities: CavitySettings, current:float, xi:float, coupling:bool) -> dict:
    
    # Create output
    output = {'current': [current], 'beta_harm': [cavities.beta_harm]}
    
    # ---- Step 1 ----
        
    # Calculate cavity phases and harmonic cavity voltage
    phase_main, phase_harm = phases_symmetric_profile(cavities.V_main, cavities.n_harm, xi, machine.energy_loss)        
    V_harm = - xi*cavities.V_main*np.sin(phase_main)/(cavities.n_harm*np.sin(phase_harm))
    
    output['phase_main'] = phase_main
    output['phase_harm'] = phase_harm
    output['V_harm'] = V_harm
        
    # Check so equilibrium phase exist
    if equilibrium_phase_exists(phase_main):
        output['equilibrium_phase'] = True
    else:
        output['equilibrium_phase'] = False
        return output
            
    # ---- Step 2 ---- 
       
    # Calculate the coefficients of the synchrotron potential
    a,b,c = potential_coefficients(machine.energy,machine.rev_time,machine.mom_comp,cavities.rf_freq,cavities.V_main,phase_main,cavities.n_harm,V_harm,phase_harm)
    
    output['a'] = a
    output['b'] = b
    output['c'] = c
    
    # ---- Step 3 ----  
      
    # Calculate the bunch length and form factors 
    b_length = bunch_length(machine.mom_comp,machine.energy_spread,a,b,c,10000e-12) # How should be the range be chosen?        
    formfactor_main,formfactor_harm = form_factors(cavities.rf_freq,cavities.n_harm,b_length)
    
    output['bunch_length'] = b_length 
    output['form_factor_main'] = formfactor_main
    output['form_factor_harm'] = formfactor_harm
      
    # ---- Step 4 ---- 
      
    # Calculate the tuning angles for operation in the compensated condition
      
    # Loaded cavity parameters
    R_main_loaded = cavities.R_main/(1+cavities.beta_main)
    Q_main_loaded = cavities.Q_main/(1+cavities.beta_main)
    
    R_harm_loaded = cavities.R_harm/(1+cavities.beta_harm)
    Q_harm_loaded = cavities.Q_harm/(1+cavities.beta_harm)
    
    # Cavity tunings
    tuning_angle_main = compensated_condition(cavities.V_main,phase_main,formfactor_main,R_main_loaded,current)
    tuning_angle_harm = compensated_condition(V_harm,phase_harm,formfactor_harm,R_harm_loaded,current)
    
    output['R_main_loaded'] = R_main_loaded
    output['R_harm_loaded'] = R_harm_loaded
    output['Q_main_loaded'] = Q_main_loaded
    output['Q_harm_loaded'] = Q_harm_loaded
    output['tuning_angle_main'] = tuning_angle_main
    output['tuning_angle_harm'] = tuning_angle_harm   

    # ---- Step 5 ----
    
    # ---- Estimate the robinson frequency ----
    omega_r = estimated_robinson_freq(machine,cavities,xi,formfactor_main,formfactor_harm,phase_main)    
    output['omega_r'] = omega_r
    
    # ---- Zero frequency instability ----
    if coupling:
        zero_frequency_instability_mode = zero_frequency_instability_coupling(machine,cavities,current,b_length,R_main_loaded,Q_main_loaded,formfactor_main,tuning_angle_main,R_harm_loaded,Q_harm_loaded,formfactor_harm,tuning_angle_harm,omega_r)
    else:
       zero_frequency_instability_mode = zero_frequency_instability_no_coupling(machine,cavities,current,b_length,R_main_loaded,formfactor_main,tuning_angle_main,R_harm_loaded,formfactor_harm,tuning_angle_harm,omega_r)
                 
    # If zero frequency instability exists return without calculating AC instability        
    if zero_frequency_instability_mode > 0:        
        return output
        
    # ---- Landau threshold ----
    landau = landau_threshold(machine, omega_r, c, b)
    output['landau_threshold'] = landau

    # ---- AC Robinson instabilities ----
    if coupling:
        ac_robinson_exist = ac_robinson_coupling(machine, cavities, current, b_length, R_main_loaded, Q_main_loaded, formfactor_main,
                                            tuning_angle_main, R_harm_loaded, Q_harm_loaded, formfactor_harm, tuning_angle_harm, omega_r, landau)
    else:
        ac_robinson_exist = ac_robinson_no_coupling(machine, cavities, current, b_length, R_main_loaded, Q_main_loaded, formfactor_main,
                                        tuning_angle_main, R_harm_loaded, Q_harm_loaded, formfactor_harm, tuning_angle_harm, omega_r, landau)   
    
    output['ac_robinson_exist'] = ac_robinson_exist

    return output