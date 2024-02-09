# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 13:57:11 2024

Script to run formation flight data collection
@author: coe-windtunnel
"""

import numpy as np
import pandas as pd

import time
import wt_formation_flight_env as form_flight_env

def run():
    freq = 1000
    num = 5000
    
    
    
    test_or_tare = input("Is this a tare? (Y/N): ")
    Re = '100k' #reynolds number
    #Re = input("Reynolds number: ")
    #aoa = input("Angle of attack: ")
    aoa = '5'
    if test_or_tare in ['N','n']:
        xpos = '1'
        #xpos = input("X position: ")
        ypos = input("Y position [0-6]: ")
        zpos = input("Z position [-1,0,1,2]: ")
        it = input("iteration [0-2]: ")
        t_o_t = 'Test'
        filename = 'Re'+Re+'_AoA'+aoa+'_X'+xpos+'_Y'+ypos+'_Z'+zpos+'_iter'+it+'_'+t_o_t
    else:
        xpos = '1'
        #xpos = input("X position: ")
        zpos = input("Z position [-1,0,1,2]: ")
        it = input("iteration [0-2]: ")
        t_o_t = 'Tare'
        filename = 'form_flight_Re'+Re+'_AoA'+aoa+'_X'+xpos+'_Z'+zpos+'_iter'+it+'_'+t_o_t
    
    f_dir = t_o_t+'/'
    
    print('Proceeding with new',t_o_t,',saved under: ',f_dir+filename+'.csv')
    env = form_flight_env(freq, num, filename, f_dir)
    _ = env.get_data()
    
    
    
    
    
    
#%% 
if __name__ == "__main__":
    run()
    print('done')