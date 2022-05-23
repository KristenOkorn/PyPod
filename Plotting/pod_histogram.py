# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 14:07:03 2022

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

#loop through for each location
for i in range(5):
    #get the current pod
    currentpod = pods[i]
        
    #get the filename to be loaded
    filename = "YPOD{}.xlsx".format(currentpod)
    #combine the paths with the filenames for each
    filepath = os.path.join(path, filename)
    #load in the matlab data from excel
    mat = pd.read_excel(filepath, index_col=0)  

    #create an array with the location repeating
    place=np.repeat(currentpod, len(mat))
    #add the array to the individual dataframe
    mat['Pod'] = place.tolist()
    #add this location to the overall dataframe
    newdata = newdata.append(mat,ignore_index=True)
    
#Creating axes & histogram
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
sns.set_style("whitegrid")
ax = sns.histplot(newdata, x="CH4", hue="Pod", element="step")
plt.xlim(0, 10)

#add string + variable for y axis, plus limits
xlabel = 'Methane (ppm)'
plt.xlabel(xlabel)

#add string + variable for plot title
titl = 'Landfill Methane'
plt.title(titl)
   
#final plotting & saving
imgname = 'CH4podhistogram.png'
imgpath = os.path.join(path, imgname)
plt.savefig(imgpath)
plt.show()