#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 12:15:04 2023

@author: Teresia Olsson, teresia.olsson@helmholtz-berlin.de
"""

from types import SimpleNamespace  

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
    
    # Turn data into namespace to easier access the parameters
    data = SimpleNamespace(**params)
    
    return data



