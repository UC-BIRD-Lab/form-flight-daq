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
from nidaqmx.constants import TerminalConfiguration
from nidaqmx.stream_readers import AnalogSingleChannelReader

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
        
        self.press_reader = AnalogSingleChannelReader(press_taps.in_stream)
        
        # initialize data array
        self.press_read_array = numpy.zeros(self.num_samp, dtype=numpy.float64)

        # acquire and store in read_array
        self.press_reader.read_many_sample(
            read_array, number_of_samples_per_channel=self.num_samp,
            timeout=10.0)
        
        # Set up communication for Pitot tube
        self.pitot = nidaqmx.Task()
        self.pitot.ai_channels.add_ai_voltage_chan("Dev3/ai25", terminal_config = TerminalConfiguration.RSE)

        self.load_cell = nidaqmx.Task()
        self.load_cell.ai_channels.add_ai_voltage_chan("Dev3/ai0")
        self.load_cell.ai_channels.add_ai_voltage_chan("Dev3/ai1")
        self.load_cell.ai_channels.add_ai_voltage_chan("Dev3/ai2")
        self.load_cell.ai_channels.add_ai_voltage_chan("Dev3/ai3")
        self.load_cell.ai_channels.add_ai_voltage_chan("Dev3/ai4")
        self.load_cell.ai_channels.add_ai_voltage_chan("Dev3/ai5")
        
        self.tap_cols = ['tap0','tap1','tap2','tap3','tap4','tap5','tap6',
                         'tap7','tap8','tap9','tap10','tap11','tap12',
                         'tap13','tap14','tap15','tap16','tap17','tap18','tap19']
        self.pitot_cols = ['dyn_press']
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
        
        taps_hist = []
        force_hist = []
        WT_flow_hist = []
        time_hist = []
        self.pause_for_timing()
        t_0 = self.prev_time
        steps = 0
        while not self.done:
            taps_ = np.array(self.press_taps.read())
            force_ = np.array(self.load_cell.read())
            WT_flow_ = np.array(self.pitot.read())
      
            taps_hist.append(taps_)
            force_hist.append(force_)
            WT_flow_hist.append(WT_flow_)
            time_hist.append(self.prev_time-t_0)
            self.pause_for_timing()
            steps += 1 
            if steps >= self.num_steps:
                self.end()
            
            
        taps_df = pd.DataFrame(taps_hist, columns=self.tap_cols)
        taps_df['time'] = time_hist
        taps_df.to_csv(self.fdir+'taps_'+self.filename+'.csv', sep=',')
        force_df = pd.DataFrame(force_hist, columns=self.force_cols)
        force_df['time'] = time_hist
        force_df.to_csv(self.fdir+'forces_'+self.filename+'.csv', sep=',')
        WT_flow_df = pd.DataFrame(WT_flow_hist, columns=self.pitot_cols)
        WT_flow_df['time'] = time_hist
        WT_flow_df.to_csv(self.fdir+'pitot_'+self.filename+'.csv', sep=',')
        print("Files Saved")
            
        return taps_df, force_df, WT_flow_df

    
    
    
    def end(self):
        self.done = True
        self.load_cell.close()
        self.press_taps.close()
        self.pitot.close()

        

    