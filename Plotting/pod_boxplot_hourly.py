# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 13:39:44 2022

Creates a separate boxplot for each pod where the x-axis is hour of day

@author: okorn
"""

#Import helpful toolboxes etc
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import scipy.io

#Need to get the pod data in timetable, retimed format
#Load this data into excel first

#create a directory path for us to pull from / save to
path = 'C:\\Users\\okorn\\Documents\\FRAPPE\\Pandora\\'
    
#initialize a dataframe to add the concentrations of each to
newdata = pd.DataFrame()

#set up pod structures to loop through
pods = ['D4','F3','F1','F4','F9']

#add the matching locations for each pod (same order as above)
location = ['BAO','BAO','NREL','Platteville','Platteville']

#pollutant loop first

#set up pollutant structures to loop through
pollutants = ['O3','NO2']

#loop through for each pollutant
for n in range(len(pollutants)):
    
    #loop through for each pod
    for i in range(len(pods)):
        #get the current pod
        currentpod = pods[i]
        
        #get the filename to be loaded
        filename = "YPOD{}_{}.csv".format(currentpod,pollutants[n])
        #combine the paths with the filenames for each
        filepath = os.path.join(path, filename)
        #load in the matlab data from excel
        mat = pd.read_csv(filepath, index_col=0)  

        #remove negatives (if needed)
        mat = mat[(mat['{}'.format(pollutants[n])]>0)]
        
        #make sure the index is a datetime
        mat.index = pd.to_datetime(mat.index)
    
        #create a new column to add our hour of day to
        mat["hours"] = pd.NaT
        #get the hours alone
        hours = mat.index.hour
        #convert to pacific daylight time
        hours = hours -7
        #now correct the negatives
        hours = np.where(hours < 0, hours + 24 , hours)

    #Creating axes & histogram
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    sns.set_color_codes(palette='pastel')
    sns.set_style("whitegrid")
    ax = sns.boxplot(y=mat['{}'.format(pollutants[n])],x=hours,color = 'r')

    #add string + variable for y axis & plot title
    if pollutants[n] == 'O3':
        y = '$O_3$ (ppb)'
        titl = "INSTEP $O_3$ - {}".format(location[i])
    else:
        y = '$NO_2$ (ppm)'
        titl = "{} $NO_2$".format(location[i])
    plt.ylabel(y)
    
    #add string + variable for x axis, plus limits
    x = 'Hour of Day (PDT)'
    plt.xlabel(x)
    plt.title(titl)
        
    #final plotting & saving
    imgname = '{}_{}_boxplot_hourly.png'.format(pollutants[n],location[i])
    imgpath = os.path.join(path, imgname)
    plt.savefig(imgpath)
    plt.show()