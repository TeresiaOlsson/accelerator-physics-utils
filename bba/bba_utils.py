#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 11:32:39 2023

@author: Teresia Olsson, teresia.olsson@helmholtz-berlin.de
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import at

#plt.ioff() # Turn of showing plots


def find_device(ring,famName,device):        
    elements = ring[famName]       
    for elem in elements:           
        try:
            if str(elem.Device) == device:
                return elem
        except:
            continue

def bpm_offset(measurement:str, qms:np.ndarray, figure_dir, make_plots):
    """
    Calculate BPM offsets from measurement of a single quad in specific plane.
    The offset is found by doing a least-squares fit and then taking the mean
    of the offset which minimizes the orbit distortion in each BPM.

    Parameters
    ----------
    measurement : str
        Quad name and plane for measurement
    qms : np.ndarray
        QMS

    Returns
    -------
    mean_offset : TYPE
        Output offset

    """
    
    # Find index of the BPM closes to the quad in the BPM device list
    bpm_index = np.where(np.all(qms["BPMDevList"][0][0] == qms["BPMDev"][0][0], axis=1))[0][0]
    
    # Quad name
    quad = measurement.split("_")[0]
    plane_name = measurement.split("_")[1]
        
    # Select the plane
    if qms["QuadPlane"] == 1:
        plane = 'x'
        x0 = qms["x0"][0][0]
        x1 = qms["x1"][0][0]
        x2 = qms["x2"][0][0]
    else:
        plane = 'y'
        x0 = qms["y0"][0][0]
        x1 = qms["y1"][0][0]
        x2 = qms["y2"][0][0]
        
    # Fit the data
    
    x = (x1[bpm_index,:] + x2[bpm_index,:])/2     
             
    [offsets, fit_result] = fit_data(x,x1,x2)
    
    # Remove the bad BPMs before calculating the mean offset
    # BPMs are removed for 1. Bad Status, 2. BPM Outlier, 3. Small Slope, or 4. Center Outlier
    bad_indices = np.zeros(len(offsets),dtype=bool)
    
    # 1. Bad status
    index_status = np.where(qms["BPMStatus"][0][0].squeeze() == 0 )
    bad_indices[index_status] = True
        
    # 2. BPM outliers
    # No BPM std exist 

    # 3. Small slope (remove BPM if the slope  is less than MinSlopeFraction * the maximum slope)
    MinSlopeFraction = .25
    slopes = np.absolute(fit_result[:,1])
    slopes = np.sort(slopes)
    slopes = slopes[int(np.round(len(slopes)/2)):int(len(slopes))] # Remove the first half
    if len(slopes) > 5:
        slopesMax = slopes[-1-4]
    else:
        slopesMax = slopes[-1]
    index_slopes = np.where( np.absolute(fit_result[:,1]) < (slopesMax * MinSlopeFraction))
    bad_indices[index_slopes] = True
      
    # 4. Center outlier (remove BPM when offset - mean(offset) > 1 std)
    CenterOutlierFactor = 1
    
    total_indices = np.linspace(0,len(offsets)-1,len(offsets))
#    index_ok = np.copy(total_indices)
    
    offset1 = offsets[~bad_indices]  # Remove the offsets that already is known to be bad
    index_ok = total_indices[~bad_indices]
    
    index = np.where( np.absolute(offset1 - np.mean(offset1)) > (CenterOutlierFactor * np.std(offset1,ddof=1))  ) 
    index_center = index_ok[index]
    bad_indices[index_center.astype(int)] = True
       
    # Remove the outliers from the offsets and fit result
    offsets = offsets[~bad_indices]
    fit_result = fit_result[~bad_indices,:]
    
    # Calculate mean offset
    offset = np.mean(offsets)
       
    # Plot 
    if make_plots:
        difference_orbit = np.transpose(x2-x1)    
        
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(x,difference_orbit,'-x')
        ax1.set_title('Quad: ' + quad)
        ax1.set_ylabel(r"$\Delta$ BPM{} [mm]" "\n" "(raw)".format(plane))
        ax1.set_xlabel("BPM{}[{},{}] [mm]".format(plane,qms["BPMDev"][0][0][0][0],qms["BPMDev"][0][0][0][1]))
         
        # Plot the fitted data
        xx = np.linspace(x[0], x[-1], 200)
        for i in range(len(fit_result)):
            y = fit_result[i,0] + fit_result[i,1]*xx
            ax2.plot(xx,y)
                     
        # Fix layout, save the fig and close it
        ax2.set_ylabel(r"$\Delta$ BPM{} [mm]" "\n" "(LS fit)".format(plane))
        ax2.set_xlabel("BPM{}[{},{}] [mm]".format(plane,qms["BPMDev"][0][0][0][0],qms["BPMDev"][0][0][0][1])) 
        plt.tight_layout()
        plt.savefig("{}/{}_{}_bpm_offset.png".format(figure_dir,quad,plane_name))
        plt.close(fig)
      
    return offset*1e-3 # Change to m instead of mm

def fit_data(x, x1, x2):
    """
    
    Fit the BPM offset for a single BPM.

    Parameters
    ----------
    x : TYPE
        Mean orbit at the BPM.
    x1 : TYPE
        Orbit for first quad change in all BPMs.
    x2 : TYPE
        Orbit for seconds quad change in all BPMs.

    Returns
    -------
    offsets : TYPE
        Offsets calculated for each least-squares fit
    fit_results : TYPE
        Output 

    """    
    # Set up and perform least-squares fit
    A = np.concatenate([np.ones((len(x),1)), x.reshape([-1,1])],axis=1)
    y = np.transpose(x2 - x1)
    [fit_results, residues, rank, s] = scipy.linalg.lstsq(A,y)
    
    # Calculate x where the lines intersect with y = 0
    intersects = - fit_results[0,:] / fit_results[1,:]
    
    #mean_offset = np.mean(offset)
        
    return intersects, np.transpose(fit_results)


    # merit_function = x2 - x1 # y

    # #N = np.shape(x1)[1] 
    # X = np.concatenate([np.ones((len(x),1)), x.reshape([-1,1])],axis=1) # X
    
    # invXX   = np.linalg.inv(np.transpose(X) @ X) # (X^T * X)^-1
    # invXX_X = invXX @ np.transpose(X) # (X^T * X)^-1 * X^T
    
    # # Make a linear fit for each BPM
    # bhat = np.zeros( (len(merit_function),2) )
    # offset = np.zeros( (len(merit_function),1) )
    
    # for i in range(len(merit_function)):
    #     b = invXX_X @ merit_function[i,:]
    #     bhat[i,:] = b
                
    #     # Calculate BPM offset (given by x where the lines intersect with y = 0)
    #     offset[i,:] = -b[0]/b[1]
    
    # # Calculate mean off offset
    # mean_offset = np.mean(offset)
    
def steerer_change(measurement:str, qms:np.ndarray, figure_dir, make_plots):
    
    # Find steerer
    steerer = qms["CorrDevList"][0][0][0]
       
    # Quad name
    quad = measurement.split("_")[0]
    plane_name = measurement.split("_")[1]
        
    # Select the plane
    if qms["QuadPlane"] == 1:
        plane = 'x'
        x0 = qms["x0"][0][0]
        x1 = qms["x1"][0][0]
        x2 = qms["x2"][0][0]
    else:
        plane = 'y'
        x0 = qms["y0"][0][0]
        x1 = qms["y1"][0][0]
        x2 = qms["y2"][0][0]
        
    # Fit the data
    nSteps = qms["NumberOfPoints"][0][0][0][0]
    CorrDelta = qms["CorrDelta"][0][0][0][0]
    steerer_setting = np.linspace(-CorrDelta,CorrDelta,nSteps)    
               
    [changes, fit_result] = fit_data(steerer_setting,x1,x2)
    
    # Remove the bad BPMs before calculating the mean offset
    # BPMs are removed for 1. Bad Status, 2. BPM Outlier, 3. Small Slope, or 4. Center Outlier
    bad_indices = np.zeros(len(changes),dtype=bool)
    
    # 1. Bad status
    index_status = np.where(qms["BPMStatus"][0][0].squeeze() == 0 )
    bad_indices[index_status] = True
        
    # 2. BPM outliers
    # No BPM std exist 

    # 3. Small slope (remove BPM if the slope  is less than MinSlopeFraction * the maximum slope)
    MinSlopeFraction = .25
    slopes = np.absolute(fit_result[:,1])
    slopes = np.sort(slopes)
    slopes = slopes[int(np.round(len(slopes)/2)):int(len(slopes))] # Remove the first half
    if len(slopes) > 5:
        slopesMax = slopes[-1-4]
    else:
        slopesMax = slopes[-1]
    index_slopes = np.where( np.absolute(fit_result[:,1]) < (slopesMax * MinSlopeFraction))
    bad_indices[index_slopes] = True
      
    # 4. Center outlier (remove BPM when offset - mean(offset) > 1 std)
    CenterOutlierFactor = 1
    
    total_indices = np.linspace(0,len(changes)-1,len(changes))
#    index_ok = np.copy(total_indices)
    
    changes1 = changes[~bad_indices]  # Remove the offsets that already is known to be bad
    index_ok = total_indices[~bad_indices]
    
    index = np.where( np.absolute(changes1 - np.mean(changes1)) > (CenterOutlierFactor * np.std(changes1,ddof=1))  ) 
    index_center = index_ok[index]
    bad_indices[index_center.astype(int)] = True
       
    # Remove the outliers from the changes and fit result
    changes = changes[~bad_indices]
    fit_result = fit_result[~bad_indices,:]
    
    # Calculate mean change
    change = np.mean(changes)
    
    if make_plots:

        difference_orbit = np.transpose(x2-x1)    
        
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(steerer_setting,difference_orbit,'-x')
        ax1.set_title('Quad: ' + quad)
        ax1.set_ylabel(r"$\Delta$ BPM{} [mm]" "\n" "(raw)".format(plane))
        ax1.set_xlabel(r"Steerer $\Delta$ [{},{}]{} [A]".format(steerer[0],steerer[1],plane))
        
        # Plot the fitted data
        xx = np.linspace(steerer_setting[0], steerer_setting[-1], 200)
        for i in range(len(fit_result)):
            y = fit_result[i,0] + fit_result[i,1]*xx
            ax2.plot(xx,y)
                         
        # Fix layout, save the fig and close it
        ax2.set_ylabel(r"$\Delta$ BPM{} [mm]" "\n" "(LS fit)".format(plane))
        ax2.set_xlabel(r"Steerer $\Delta$ [{},{}]{} [A]".format(steerer[0],steerer[1],plane))
        plt.tight_layout()
        plt.savefig("{}/{}_{}_steerer_change.png".format(figure_dir,quad,plane_name))
        plt.close(fig)
    
    return change

def orbit_change(ring, plane, steerer_name, steerer_change, refpts_name):
    
    # Get refpts in lattice from name   
    refpts = [False] * (len(ring)+1)
    for elem in refpts_name:
        refpts += ring.get_bool_index(elem.lower())
    
    # Find the corrector to change
    corr = ring[steerer_name.lower()][0]   
    
    if type(corr) is not at.Corrector: # If the corrector is drift change it to a corrector
        index = ring.get_bool_index(steerer_name.lower())       
        ring[index] = at.Corrector(corr.FamName, corr.Length, [0, 0])
        corr = ring[steerer_name.lower()][0]
    
    # Get orbit before kick
    [_, orbit0] = ring.find_orbit(refpts)

    # Apply the kick, get the orbit after kick and calculate the orbit change
    orbit_change = []
    brho = 5.67229 # This number comes from MML
    edf = 1/brho
        
    for setting in steerer_change:
        if plane == 'Hor':        
            if steerer_name[0:3] == "HS1":
                conversion_factor = -0.00553438*edf
                kick = conversion_factor*setting
                corr.KickAngle = [kick, 0] 
            elif steerer_name[0:3] == "HS4":
                conversion_factor = -0.0044502*edf
                kick = conversion_factor*setting 
                corr.KickAngle = [kick, 0]        
        else:       
            conversion_factor = 0.00297607*edf
            kick = conversion_factor*setting  
            corr.KickAngle = [0, kick]
                
        # Get orbit after the kick
        [_, orbit1]  = ring.find_orbit(refpts)
                 
        # Calculate the orbit change 
        diff_orbit = orbit1 - orbit0        
        if plane == 'Hor':
            orbit_change.append(diff_orbit[:,0])
        else:
            orbit_change.append(diff_orbit[:,2])
                 
        # Restore corrector
        corr.KickAngle = [0,0]
        
    return np.transpose(np.array(orbit_change))
        
        
def quad_offset(ring,quad_name,plane,steerer_name,change):
    
    return orbit_change(ring, plane, steerer_name, [change], [quad_name])[0][0] # Change the sign to get the offset 

def distorted_orbit(ring, quad_name, plane, kick, refpts_name):
        
    # Find index of the quad
    quad_index = np.where( ring.get_bool_index(quad_name.lower()) == True)[0][0]
            
    # Insert a corrector before the quad
    ring_copy = ring.copy() # Make copy of the ring to not change the original object
    temp_corr = at.Corrector("temp_corr",0.0,[0, 0])
    ring_copy.insert(quad_index,temp_corr)
    
    # Get refpts in lattice from name   
    refpts = [False] * (len(ring_copy)+1)
    for elem in refpts_name:
        refpts += ring_copy.get_bool_index(elem.lower())
    
    orbit_change = []
    # Get orbit before kick
    [_, orbit0] = ring_copy.find_orbit(refpts)
    
    # Apply the kick, get the orbit after kick and calculate the orbit change
    if plane == 'Hor':        
        temp_corr.KickAngle = [kick, 0]
    else:
        temp_corr.KickAngle = [0, kick]
    
    # Get orbit after the kick
    [_, orbit1]  = ring_copy.find_orbit(refpts)
                 
    # Calculate the orbit change 
    diff_orbit = orbit1 - orbit0        
    if plane == 'Hor':
        orbit_change.append(diff_orbit[:,0])
    else:
        orbit_change.append(diff_orbit[:,2])    
    
    return np.transpose(np.array(orbit_change))
        
  
# def quad_offset(ring,quad_name,plane,steerer_name,change):
    
#     # Get orbit at quad before kick    
#     [_, orbit0] = ring.find_orbit(quad_name.lower())
    
#     corr = ring[steerer_name.lower()][0]
    
#     # If the corrector is drift change it to a corrector
#     if type(corr) is not at.Corrector:
#         index = ring.get_bool_index(steerer_name.lower())       
#         ring[index] = at.Corrector(corr.FamName, corr.Length, [0, 0])
#         corr = ring[steerer_name.lower()][0]
        
#     brho = 5.67229 # Fix this calculation
#     edf = 1/brho
    
#     # Apply kick
#     if plane == 'Hor':        
#         if steerer_name[0:3] == "HS1":
#             conversion_factor = -0.00553438*edf
#             kick = conversion_factor*change
#         elif steerer_name[0:3] == "HS4":
#             conversion_factor = -0.0044502*edf
#             kick = conversion_factor*change  
#         corr.KickAngle = [kick, 0]        
#     else:       
#         conversion_factor = 0.00297607*edf
#         kick = conversion_factor*change      
#         corr.KickAngle = [0, kick]
                
#     # Check the orbit at the quad 
#     [_, orbit1]  = ring.find_orbit(quad_name.lower())
     
#     # Calculate the difference orbit at the quad
#     diff_orbit = orbit1 - orbit0
     
#     # Restore corrector
#     corr.KickAngle = [0,0]
             
#     if plane == 'Hor':
#         return -diff_orbit[0][0]
#     else:
#         return -diff_orbit[0][2] 

# def orbit_change(ring, plane, steerer_name, steerer_settings, bpm_names):
    
#     # Find BPMs
#     bpm_pos = [False] * (len(ring)+1)
    
#     for bpm in bpm_names:
#         bpm_pos += ring.get_bool_index(bpm.lower())
        
#     corr = ring[steerer_name.lower()][0]
    
#     # If the corrector is drift change it to a corrector
#     if type(corr) is not at.Corrector:
#         index = ring.get_bool_index(steerer_name.lower())       
#         ring[index] = at.Corrector(corr.FamName, corr.Length, [0, 0])
#         corr = ring[steerer_name.lower()][0]
      
#     brho = 5.67229
# #    brho = 5.670589
#     edf = 1/brho
    
#     orbit = []
              
#     for setting in steerer_settings:
        
#         # Get orbit at BPMs before kick    
#         [_, orbit0] = ring.find_orbit(bpm_pos)
        
#         # Apply kick
#         if plane == 'Hor':        
#             if steerer_name[0:3] == "HS1":
#                 conversion_factor = -0.00553438*edf
#                 kick = conversion_factor*setting
#                 #kick = 2*conversion_factor*setting
#             elif steerer_name[0:3] == "HS4":
#                 conversion_factor = -0.0044502*edf
#                 kick = conversion_factor*setting 
#                 #kick = 2*conversion_factor*setting
#             corr.KickAngle = [kick, 0]        
#         else:       
#             conversion_factor = 0.00297607*edf
#             kick = conversion_factor*setting  
#             #kick = 2*conversion_factor*setting 
#             corr.KickAngle = [0, kick]
                    
#         # Check the orbit at the BPMs
#         [_, orbit1]  = ring.find_orbit(bpm_pos)
         
#         # Calculate the difference orbit at the BPMs
#         diff_orbit = orbit1 - orbit0
        
#         if plane == 'Hor':
#             orbit.append(diff_orbit[:,0])
#         else:
#             orbit.append(diff_orbit[:,2])
                 
#         # Restore corrector
#         corr.KickAngle = [0,0]
        
#     return np.transpose(np.array(orbit))
        

# # Conversion factors MML
# vcm = 0.00297607 * edf
# # Horizontal
# hbm = -0.00159919 * edf
# hs1 = -0.00553438 * edf
# hs4 = -0.0044502 * edf

# edf = 1/getbrho(1.7) = 0.17629552040612
# getbrho = 5.67229
    
    # Which quad belongs to which steerer
    # Conversion factor between steerer current and kick
    # Which change in orbit at quad/BPM for given kick -> compare to model
    