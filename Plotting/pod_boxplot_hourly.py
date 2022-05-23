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
path = 'C:\\Users\\okorn\\Documents\\NASA Postdoc\\Practice Pod Data\\'

    
#initialize a dataframe to add the concentrations of each to
newdata = pd.DataFrame()

#set up pod structures to loop through
pods = ['B2','B3','B5','C3','G6','G7']

#loop through for each pod
for i in range(5):
    #get the current pod
    currentpod = pods[i]
        
    #get the filename to be loaded
    filename = "YPOD{}.xlsx".format(currentpod)
    #combine the paths with the filenames for each
    filepath = os.path.join(path, filename)
    #load in the matlab data from excel
    mat = pd.read_excel(filepath, index_col=0)  

    
    #create a new column to add our hour of day to
    mat["hours"] = pd.NaT
    #get the hours alone
    hours = mat.index.hour

    
    #Creating axes & histogram
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    sns.set_style("whitegrid")
    ax = sns.boxplot(y=mat["CH4"], x=hours)
    #plt.xlim(0, 10)

    #add string + variable for y axis, plus limits
    y = 'Methane (ppm)'
    plt.ylabel(y)
    
    #add string + variable for x axis, plus limits
    x = 'Hour of Day'
    plt.xlabel(x)

    #add string + variable for plot title
    titl = '{} Landfill Methane'.format(currentpod)
    plt.title(titl)
   
    #final plotting & saving
    imgname = 'CH4_{}_boxplot_hourly.png'.format(currentpod)
    imgpath = os.path.join(path, imgname)
    plt.savefig(imgpath)
    plt.show()