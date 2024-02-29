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
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx.stream_readers import AnalogSingleChannelReader


class ff_env():

    def __init__(self, freq=1000, num=5000):
        # specify the channel that holds the pitot information
        self.pitot_pressure_channel = "Dev3/ai0"

        self.num_samp = num
        self.freq = freq
        
        # Set up ni communication for just the pitot tube readings
        self.pitot_data = nidaqmx.Task()
        self.pitot_data.ai_channels.add_ai_voltage_chan(
            self.pitot_pressure_channel,
            terminal_config=TerminalConfiguration.RSE
            )
        self.pitot_data.timing.cfg_samp_clk_timing(self.freq)
        self.pitot_reader = AnalogSingleChannelReader(self.pitot_data.in_stream)

        # Set up ni communication for all pressure taps
        self.press_taps = nidaqmx.Task()
        # read certain channels
        first_pressure_channel = 0
        last_pressure_channel = 20
        for channel_number in range(first_pressure_channel, last_pressure_channel+1):
            channel_string = f"Dev3/ai{channel_number}"
            self.press_taps.ai_channels.add_ai_voltage_chan(
                channel_string,
                terminal_config=TerminalConfiguration.RSE
            )
        # sample pressure taps at specified frequency
        self.press_taps.timing.cfg_samp_clk_timing(self.freq)
        # reads nidaq information for all pressure reader
        self.press_reader = AnalogMultiChannelReader(self.press_taps.in_stream)

        # initialize data arrays for pressure data
        self.press_read_array = np.zeros((21, self.num_samp), dtype=np.float64)
        self.pitot_press_read_array = np.zeros(
            (1, self.num_samp), dtype=np.float64)

        # Set up separate task for load cell readings
        self.load_cell = nidaqmx.Task()
        self.load_cell.ai_channels.add_ai_voltage_chan("Dev2/ai0")
        self.load_cell.ai_channels.add_ai_voltage_chan("Dev2/ai1")
        self.load_cell.ai_channels.add_ai_voltage_chan("Dev2/ai2")
        self.load_cell.ai_channels.add_ai_voltage_chan("Dev2/ai3")
        self.load_cell.ai_channels.add_ai_voltage_chan("Dev2/ai4")
        self.load_cell.ai_channels.add_ai_voltage_chan("Dev2/ai5")
        self.load_cell.timing.cfg_samp_clk_timing(self.freq)
        self.force_reader = AnalogMultiChannelReader(self.load_cell.in_stream)
        # initialize data arrays for force readings
        self.force_read_array = np.zeros((6, self.num_samp), dtype=np.float64)

        # label columns for data recording
        self.tap_cols = ['pitot', 'tap0', 'tap1', 'tap2', 'tap3', 'tap4', 'tap5', 'tap6',
                         'tap7', 'tap8', 'tap9', 'tap10', 'tap11', 'tap12',
                         'tap13', 'tap14', 'tap15', 'tap16', 'tap17', 'tap18', 'tap19']
        self.force_cols = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5']

        # overall status of the run
        self.done = False

    def get_pitot_voltage(self):
        # acquire pitot tube pressure information
        pitot_data_stream = self.pitot_data.read()
        return pitot_data_stream
        
    def airspeed_from_pressure(dynamic_pressure, rho, K_constant=1):
        # default K_constant to 1 because we only use 1 pitot tube
        
        #differential_static_pressure = P_L - P_S
        V_j = np.sqrt(2*(dynamic_pressure)/(rho*K_constant)) #velocity at station J (where model mounts in test section)
        return V_j
        
    def output_velocity_stream(self):
        velocity = self.airspeed_from_pressure(
            self.press_taps
            )

    def record_data(self, fdir='Test/', filename='Test'):
        # acquire and store in read_array
        self.press_reader.read_many_sample(
            self.press_read_array,
            number_of_samples_per_channel=self.num_samp
            )
   
        self.force_reader.read_many_sample(
            self.force_read_array,
            number_of_samples_per_channel=self.num_samp,
            )
            
        taps_df = pd.DataFrame(self.press_read_array.T, columns=self.tap_cols)
        taps_df.to_csv(fdir+'taps_'+filename+'.csv', sep=',')
        force_df = pd.DataFrame(self.force_read_array.T, columns=self.force_cols)
        force_df.to_csv(fdir+'forces_'+filename+'.csv', sep=',')
     
        print("Files Saved")
        
        return taps_df, force_df

    
    def end(self):
        self.done = True
        self.load_cell.close()
        self.press_taps.close()
        
    def end_pitot(self):
        self.pitot_data.close()
     
        

    