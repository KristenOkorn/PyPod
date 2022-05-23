# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 12:48:32 2022

Create scatterplot & color points by xy wind direction

@author: okorn
"""

# -*- coding: utf-8 -*-


#Import helpful toolboxes etc
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.lines as mlines

#Need to use the pod data in timetable, retimed format
#Load this data into excel first

#create a directory path for us to pull from / save to
path = 'C:\\Users\\okorn\\Documents\\NASA Postdoc\\Practice Pod Data\\'

#set up pod structures to loop through
pods = ['B2','B3','B5','C3','G6','G7']
#select which pod we'll compare the rest to
keypod = 'G7'

#load in & clean the met data
#get the met filename
met_filename='met_1stmonth_shifted.xlsx'
#combine the paths with the filenames for each
met_filepath = os.path.join(path,met_filename)
#load in the met data from excel
wind = pd.read_excel(met_filepath, index_col=0)  
#retime the met data to hourly
wind = wind.resample('H').mean()
#remove missing values from the dataframe
wind = wind.dropna() 

#load in & format the data for the key pod
#get the filename to be loaded
filename = "YPOD{}.xlsx".format(keypod)
#combine the paths with the filenames for each
filepath = os.path.join(path, filename)
#load in the matlab data from excel
mat = pd.read_excel(filepath, index_col=0)  
#rename the column (by index) to the pod name
mat.columns.values[0] = 'YPOD{}'.format(keypod)

#loop through for each location
for i in range(5):
    #get the current pod
    currentpod = pods[i]
    
    #if the current pod is the same as our key pod, skip this iteration
    if currentpod != keypod:
        
        #get the filename to be loaded
        compfilename = "YPOD{}.xlsx".format(currentpod)
        #combine the paths with the filenames for each
        compfilepath = os.path.join(path, compfilename)
        #load in the matlab data from excel
        compmat = pd.read_excel(compfilepath, index_col=0)  
        #rename the column (by index) to the pod name
        compmat.columns.values[0] = 'YPOD{}'.format(currentpod)
        
        #combine the 3 dataframes
        full = pd.concat([mat,compmat,wind],axis=1)
    
        #trying a different plotting method
        #initialize
        fig, ax = plt.subplots()
        #plot the basic scatterplot
        scatter = ax.scatter(full["YPOD{}".format(keypod)], full["YPOD{}".format(currentpod)], c=full['XY Dir (deg)'],alpha=0.5)
        #create the 1-1 line
        line = mlines.Line2D([0, 1], [0, 1], color='red')
        transform = ax.transAxes
        line.set_transform(transform)
        ax.add_line(line)
        
        #add string + variable for x axis, plus limits
        x = '{} Methane (ppm)'.format(keypod)
        plt.xlabel(x)

        #add string + variable for y axis, plus limits
        y = '{} Methane (ppm)'.format(currentpod)
        plt.ylabel(y)

        #add string + variable for plot title
        titl = 'Landfill Methane'
        plt.title(titl)
        
        # produce a legend with the unique colors from the scatter
        legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower right", title="Wind Dir (°)")
        ax.add_artist(legend1)
    
        #final plotting & saving
        imgname = 'CH4_{}_{}_scatterplot_wind.png'.format(keypod,currentpod)
        imgpath = os.path.join(path, imgname)
        plt.savefig(imgpath)
        plt.show()