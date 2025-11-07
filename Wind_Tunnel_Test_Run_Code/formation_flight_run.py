# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 13:57:11 2024

Script to run formation flight data collection
@author: coe-windtunnel
"""

import numpy as np
import pandas as pd
import os
import csv
import keyboard
import math

import wt_formation_flight_env_reader as form_flight_env


def calc_re(rho, velocity, chord, dynamic_viscosity):

    Re = rho*velocity*chord/dynamic_viscosity

    return Re


def density(p_atm, t_atm):
    q = p_atm*3386.39/(286.9*(t_atm+273.15))
    # 1 inHg. = 3386.39 Pa
    # R = 286.9 J/kgK, Gas Constant

    return q


def calc_dynamic_viscosity(t_thermocouple):
    C = 0.00000146  # Sutherland's Curve Const. C
    S = 110.4  # Sutherland's Curve Const. S
    dynamic_viscosity = C*(t_thermocouple+273.15)**(3/2) / \
        (S+273.15+t_thermocouple)
    return dynamic_viscosity


def pressure_from_voltage(voltage):
    """
     Parameters
    ------------
        Voltage: Voltage value
    Return
    -----------
        average of pressure sample data.
    """
    # pressure = stagnation_pressure - static_pressure
    # according to BARLOW, difference in pressure is approx 0.5*rho*V**2

    # M = (Span of the transducer's pressure range)/(Span of the transducer's voltage range)
    # these values can be found on the pressure transducer's specification sheet
    pressure = ((voltage-.2)*(1244.2 + 1244.2)/(5.2 - .2) - 1244.2)
    return pressure


def airspeed_from_pressure(dynamic_pressure, rho, K_constant=1):
    # default K_constant to 1 because we only use 1 pitot tube

    # differential_static_pressure = P_L - P_S
    # velocity at station J (where model mounts in test section)
    V_j = np.sqrt(2*(dynamic_pressure)/(rho*K_constant))
    # print(V_j)

    return V_j


def output_calibration_information(runname, atm_pressure, atm_temp, chord, env, calibrate=True):
    # read tare pitot voltage information
    tare_csv = pd.read_csv(f"Tare/taps_form_flight_{runname}_Tare.csv")
    tare_pitot_voltage = np.mean(tare_csv.pitot)
    tare_pressure = pressure_from_voltage(tare_pitot_voltage)
    
    # Calculate the air density for the test day
    air_density = density(atm_pressure, atm_temp)
    # Calculate the dynamic viscosity for the test day
    dym_vis = calc_dynamic_viscosity(atm_temp)

    print('press "q" to stop calibration loop')
    print('Airspeed(mph):           Reynolds Number:')
    print()
    while calibrate == True:
        # read live voltage information from pitot channel - verify this works how we expect it to...
        live_pressure = pressure_from_voltage(env.get_pitot_voltage())

        # subtract live pressure from tare pressure to get real pressure measurement
        live_calibrated_pressure = live_pressure - tare_pressure
        airspeed = airspeed_from_pressure(
            live_calibrated_pressure, air_density)

        re = calc_re(air_density, airspeed, chord, dym_vis)

        # display reynolds number until key is pressed
        if not math.isnan(re):
            #print(round(airspeed*2.23694,2),        round(re), end='\r')
            print(f"{round(airspeed*2.23694, 2)}        {round(re)}", end='\n')

        if keyboard.is_pressed("q"):
            # Key was pressed
            # await key press
            calibrate = False
            break

    # free up the pitot channel
    env.end_pitot()


def run():
    os.chdir('C:/Users/coe-windtunnel/projects/form-flight-daq')
    freq = 1000
    num = 5000  # 5000

    is_tare = input("Is this a tare? (Y/N): ")
    atm_pressure = float(input("Atmospheric pressure in in.HG for test day?"))
    atm_temp = float(input("Temperature in Degree Celsius for test day?"))
    chord = float(input("Chord length of the testing model?"))
    Re = input("Desired Reynolds number: ")
    aoa = input("Angle of attack (use '_' not '.' for decimals'): ")
    it = input("iteration [0-2]: ")
    xpos = input("X position: ")

    env = form_flight_env.ff_env(freq, num)

    if is_tare in ['N', 'n']:
        type_of_run = 'Test'
        ypos = input("Y position [0-6]: ")
        zpos = input("Z position [-1,0,1,2]: ")
        test_runname = 'Re'+Re+'_AoA'+aoa+'_X'+xpos+'_Y'+ypos+'_Z'+zpos+'_iter'+it
        tare_runname = 'Re'+Re+'_AoA'+aoa+'_X'+xpos+'_Y-_Z-_iter'+it
        
        filename = f'form_flight_{test_runname}_{type_of_run}'
        
        # output calibration information to screen
        output_calibration_information(
            runname=tare_runname,
            atm_pressure=atm_pressure,
            atm_temp=atm_temp,
            chord=chord,
            env=env
        )
    else:
        type_of_run = 'Tare'
        tare_runname = 'Re'+Re+'_AoA'+aoa+'_X'+xpos+'_Y-_Z-_iter'+it

        filename = f'form_flight_{tare_runname}_{type_of_run}'
        
    f_dir = type_of_run+'/'

    print('Proceeding with new', type_of_run, ',saved under: ', f_dir+filename+'.csv')

    # Take data and save to filename within f_dir
    pressure_tap_V, load_cell_V = env.record_data(f_dir, filename)

    # close all channels on the DAQ
    env.end()


# %%
if __name__ == "__main__":
    run()
    print('done')
