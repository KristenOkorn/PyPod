# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 22:01:15 2022

@author: okorn
"""

#Import in necessary packages
#on first run, need to run: pip install pykrige
import numpy as np
from matplotlib import pyplot as plt
from tkinter.filedialog import askdirectory
import os
import pandas as pd
import functools

### Step 0: have some data

#Prompt user to select folder for analysis
path = askdirectory(title='Select Folder for analysis').replace("/","\\")

#Get the list of files from this directory
from os import listdir
from os.path import isfile, join
fileList = [f for f in listdir(path) if isfile(join(path, f))]

#create a dictionary to hold our data from each file
data_dict = {}

#loop through each of the files & extract only the columns we need:
  #datetime, CO2 (wet), CH4 (wet), H2O
  
#iterate over each file in the main folder
for i in range(len(fileList)-1):
    
    #Create full file path for reading file
    filePath = os.path.join(path, fileList[i])
    
    #load in the file
    temp = pd.read_csv(filePath,delim_whitespace=True,usecols=['DATE','TIME','CO2','CH4','H2O'],parse_dates=[['DATE', 'TIME']])
    
    #Save this into our data dictionary
    data_dict['{}'.format(fileList[i])] = temp
    
#concatenate all of our data into 1 array
full_data = pd.concat(data_dict.values())
    
#apply the water vapor correction for co2
full_data['CO2'] = full_data['CO2'] / (1+(-0.012*full_data['H2O'])+(-0.000267*full_data['H2O']*full_data['H2O']))
#apply the water vapor correction for ch4
full_data['CH4'] = full_data['CH4'] / (1+(-0.009823*full_data['H2O'])+(-0.000239*full_data['H2O']*full_data['H2O']))
#delete the water column
full_data = full_data.drop(columns=['H2O'])
#apply the calibration correction for co2
full_data['CO2'] = full_data['CO2'] / 0.9975
#apply the calibration correction for ch4
full_data['CH4'] = full_data['CH4'] / 1.0078

#resample (retime) the data to minutely
full_data['DATE_TIME'] = pd.to_datetime(full_data['DATE_TIME'])
full_data = full_data.set_index('DATE_TIME').resample('1Min').ffill()

#get the final data in a convenient format
CO2 = pd.DataFrame(full_data.index,full_data['CO2'])
CH4 = pd.DataFrame(full_data.index,full_data['CH4'])

#save out the final data
savePath = os.path.join(path,'CO2.csv')
CO2.to_csv(savePath)
savePath = os.path.join(path,'CH4.csv')
CH4.to_csv(savePath)
