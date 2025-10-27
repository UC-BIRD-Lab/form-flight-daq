# form-flight-daq
Formation flight data acquisition

## Contents
This repo includes two python scripts, one containing the daqmx class, the other prompts the user for testing information (AoA, X,Y,Z, iteration,...) for file naming convention.

## Purpose
Used for collecting and saving pressure tap, force sensor, and pitot tube data during formation flight experiments. 

## Reference data
Box folder containing the MATLAB files for previous wind tunnel operation: https://ucdavis.box.com/s/df16xfmk344mtkfgysrjifpcb65wsea8


## Folder: Wind Tunnel Test Run Code
This folder contains scripts for collecting wind tunnel measurement data.
**Inputs:**  
1. DAQ system connected to all pressure taps and the pitot tube  
2. DAQ system connected to the load cell  
3. Ambient environmental conditions specified in `Testing_Day_Condition`

**Output:**  
A `Raw_Data` folder containing all measured force and pressure data.

## Script: Converter
1. The `Converter_pressure_tap.py` script processes all pressure data files. It converts the raw data into calibrated pressure values, subtracts the tare value from each test measurement, and saves the results in a new folder named `Tared`.
