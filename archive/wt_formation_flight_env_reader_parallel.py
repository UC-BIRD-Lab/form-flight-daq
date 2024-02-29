# -*- coding: utf-8 -*-
"""
Spyder Editor

This File contains a script for pressure and load cell data collection.
Input: DataRate, Number, filename, file directory
Output: Saved pressure tap, force sensor, and pitot tube files  
"""


import numpy as np
import pandas as pd

import time

import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType
from nidaqmx.stream_readers import AnalogMultiChannelReader


class ff_env():

    def __init__(self, freq=1000, num=5000, filename='Test', f_dir='Test/'):
        
        self.num_samp = num
        self.freq = freq
 
        # Set up ni communication for pressure taps 
        self.press_taps = nidaqmx.Task()
        self.press_taps.ai_channels.add_ai_voltage_chan("Dev3/ai0", terminal_config = TerminalConfiguration.RSE)
        self.press_taps.ai_channels.add_ai_voltage_chan("Dev3/ai1", terminal_config = TerminalConfiguration.RSE)
        self.press_taps.ai_channels.add_ai_voltage_chan("Dev3/ai2", terminal_config = TerminalConfiguration.RSE)
        self.press_taps.ai_channels.add_ai_voltage_chan("Dev3/ai3", terminal_config = TerminalConfiguration.RSE)
        self.press_taps.ai_channels.add_ai_voltage_chan("Dev3/ai4", terminal_config = TerminalConfiguration.RSE)
        self.press_taps.ai_channels.add_ai_voltage_chan("Dev3/ai5", terminal_config = TerminalConfiguration.RSE)
        self.press_taps.ai_channels.add_ai_voltage_chan("Dev3/ai6", terminal_config = TerminalConfiguration.RSE)
        self.press_taps.ai_channels.add_ai_voltage_chan("Dev3/ai7", terminal_config = TerminalConfiguration.RSE)
        self.press_taps.ai_channels.add_ai_voltage_chan("Dev3/ai8", terminal_config = TerminalConfiguration.RSE)
        self.press_taps.ai_channels.add_ai_voltage_chan("Dev3/ai9", terminal_config = TerminalConfiguration.RSE)
        self.press_taps.ai_channels.add_ai_voltage_chan("Dev3/ai10", terminal_config = TerminalConfiguration.RSE)
        self.press_taps.ai_channels.add_ai_voltage_chan("Dev3/ai11", terminal_config = TerminalConfiguration.RSE)
        self.press_taps.ai_channels.add_ai_voltage_chan("Dev3/ai12", terminal_config = TerminalConfiguration.RSE)
        self.press_taps.ai_channels.add_ai_voltage_chan("Dev3/ai13", terminal_config = TerminalConfiguration.RSE)
        self.press_taps.ai_channels.add_ai_voltage_chan("Dev3/ai14", terminal_config = TerminalConfiguration.RSE)
        self.press_taps.ai_channels.add_ai_voltage_chan("Dev3/ai15", terminal_config = TerminalConfiguration.RSE)
        self.press_taps.ai_channels.add_ai_voltage_chan("Dev3/ai16", terminal_config = TerminalConfiguration.RSE)
        self.press_taps.ai_channels.add_ai_voltage_chan("Dev3/ai17", terminal_config = TerminalConfiguration.RSE)
        self.press_taps.ai_channels.add_ai_voltage_chan("Dev3/ai18", terminal_config = TerminalConfiguration.RSE)
        self.press_taps.ai_channels.add_ai_voltage_chan("Dev3/ai19", terminal_config = TerminalConfiguration.RSE)
        self.press_taps.ai_channels.add_ai_voltage_chan("Dev3/ai20", terminal_config = TerminalConfiguration.RSE)
        self.press_taps.timing.cfg_samp_clk_timing(self.freq,sample_mode=AcquisitionType.FINITE,
                                                   samps_per_chan=self.num_samp)
        
        
        # initialize data array
        self.press_read_array = np.zeros((21,self.num_samp), dtype=np.float64)


        self.load_cell = nidaqmx.Task()
        self.load_cell.ai_channels.add_ai_voltage_chan("Dev2/ai0",terminal_config=TerminalConfiguration.DEFAULT)
        self.load_cell.ai_channels.add_ai_voltage_chan("Dev2/ai1",terminal_config=TerminalConfiguration.DEFAULT)
        self.load_cell.ai_channels.add_ai_voltage_chan("Dev2/ai2",terminal_config=TerminalConfiguration.DEFAULT)
        self.load_cell.ai_channels.add_ai_voltage_chan("Dev2/ai3",terminal_config=TerminalConfiguration.DEFAULT)
        self.load_cell.ai_channels.add_ai_voltage_chan("Dev2/ai4",terminal_config=TerminalConfiguration.DEFAULT)
        self.load_cell.ai_channels.add_ai_voltage_chan("Dev2/ai5",terminal_config=TerminalConfiguration.DEFAULT)
        
        #self.load_cell.timing.cfg_samp_clk_timing(self.freq)
        
        self.load_cell.timing.cfg_samp_clk_timing(self.freq,
                                                  sample_mode=AcquisitionType.FINITE,
                                                   samps_per_chan=self.num_samp)
        
        self.tap_cols = ['pitot','tap0','tap1','tap2','tap3','tap4','tap5','tap6',
                         'tap7','tap8','tap9','tap10','tap11','tap12',
                         'tap13','tap14','tap15','tap16','tap17','tap18','tap19']
    
        self.force_cols = ['f0','f1','f2','f3','f4','f5']
        
        
        
        self.filename = filename
        self.fdir = f_dir
        
        #time checking
        self.delt=1/freq
        self.dwait = 0
        self.last_dt = 0
        self.prev_time = time.time()
        self.current_time = time.time()
        self.done = False
        

 
    def pause_for_timing(self):
        self.dwait += self.delt - self.last_dt
        while True:
            current_time = time.time()
            dt = current_time - self.prev_time 
            if dt >= self.dwait:
                break
        self.last_dt = dt
        self.prev_time = current_time
        

    
 

    def get_data(self):
       
    

        taps_data = self.press_taps.read(number_of_samples_per_channel=self.num_samp)
        force_data = self.load_cell.read(number_of_samples_per_channel=self.num_samp)
       
            
        taps_df = pd.DataFrame(taps_data.T, columns=self.tap_cols)
        taps_df.to_csv(self.fdir+'taps_'+self.filename+'.csv', sep=',')
        force_df = pd.DataFrame(force_data.T, columns=self.force_cols)
        force_df.to_csv(self.fdir+'forces_'+self.filename+'.csv', sep=',')
     
        print("Files Saved")
            
        return taps_df, force_df

    
    
    
    def end(self):
        self.done = True
        self.load_cell.close()
        self.press_taps.close()
        self.pitot.close()

        

    