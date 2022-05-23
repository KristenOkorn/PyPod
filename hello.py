# Attempting to re-write parts of 'Main_Pod_Code.mat' (December 2021 edition) in python

#----------------------Import in necessary toolboxes----------------------
#Import necessary numPy syntax
import numpy as np
import scipy.io
from scipy import io, integrate, linalg, signal
from scipy.sparse.linalg import eigs
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import time
from tkinter import Tk
from tkinter.filedialog import askdirectory
from datetime import datetime
import pickle
import json
import csv
import statistics as st
from scipy.signal import savgol_filter
#----------------------End of toolbox import----------------------

    
## ----------------------- Settings for analysis ----------------------- 

#Initialize a settingsSet dictionary
settingsSet = {

# Select which type of regression you would like to run
    # 0 - multivariate linear regression
    # 1 - ANN
    # If using both, regType must be 1, field & database must be false
"regType" : 1,

# Select which type of calibration you would like to run
    # 0 - Individual (each pod gets its own unique calibration model)
    # 1 - Universal: 1-Cal (one model is applied to all pods)
    # 2 - Universal: 1-Hop (one pod's estimate are used as "reference" data)
    # If either universal method: must use "zscoreSensors" in podPreprocess
"calMode" : 2,

# For universal calibration, select the pod to calibrate the rest with
"uniPod" : "YPODB2",

# Change these values to affect what portions of the code is run
"loadOldSettings" : False,  #Load a settings set structure from an old run
"convertOnly" : False,  #Stop after converting (unconverted) pod and reference data files into .mat files
"applyCal" : True,  #Apply calibrations to the pod files in the "field" folder

# These affect how datetime data is imported by function "dataExtract"
    # Custom datetime formats to use when trying to convert non-standard import datetime strings
"datetimestrs" : "%m-%d-%Y %H:%M:%S",
    # Custom date formats to use when trying to convert non-standard import date strings
"datestrings" : "%m-%d-%Y",
# Custom time formats to use when trying to convert non-standard time strings
"timestrings" : "%H:%M:%S",

# These change the smoothing of pod and reference data in function "smoothData"
    # Time in minutes to average data by for analysis
"timeAvg" : 60,

# In these lists of columns to extract, be sure that the data will have headers with one match to each entry.  Note that partial matches will be used (e.g.: a column with "CO2" would be selected if you entered "CO" below)
    # Name of columns in reference files to try to extract from the full data file
"refGas" : "CH4", #Only use one at a time
    # List of sensor signals to extract from the full pod data file
    # Use "allcol" to extract all columns that are not dates
    # Light VOC - 'Fig2600', Heavy VOC - 'Fig2602', CO2 - 'CO2', Ozone - 'E2VO3'
    # CO with small green board (baseline mocon pin) - 'blMocon'
    # CO on larger red board (quadstats) - 'Quad_Main3'
"podSensors" : ["Fig2600","Fig2602"],
    # List of environmental sensor signals to extract from the full pod data file
    # These are separated from the other sensors in case they need to be treated differently than the other sensors (e.g. when determining interaction terms
"envSensors" : ["temperature","humidity"],

# For function lists, enter function names in the order in which you'd like them to be performed (if it matters)
    # Preprocessing functions for reference data
"refPreProcess" : ["remove999","removeZeros","removeNaNs","refSmooth"],
    # Preprocessing functions for pod data
    # If universal cal, must use "zscoreSensors"
    # If ANN, can add interaction terms here
"podPreProcess" : ["TempC2K","removeNaNs","podTimeZone","makeTElapsed","humidrel2abs","podSmooth"], #"zscoreSensors"
    # Calibration models to fit on pod data
    # If individual cal, use as many as you like
    # If universal cal, can only use one
"modelList" : "ratioSens1Te",
    # Model to use for field data (if using more than one calibration model)
    # This model must have also been called in modelList in order to work
    # Only choose one
"fieldModel" : "joannaNN",

# Validation set selection functions
# valList : "timeFold","timeofDayVal","temperatureFold","environClusterVal","concentrationsFold"
"valList" : "randVal",
    # Number of 'folds' for validation (e.g. 5 folds ~ 20% dropped for each validation set)
"nFolds" : 3,
    # Number of folds to actually use (up to nFolds) (will be used in order from 1 to nFolds)
"nFoldRep" : 3,

# Statistics to calculate
"statsList" : ["podR2","podRMSE","podMBE"], # 'podRMSE','podR2','podCorr'

# Plotting functions run during model fitting loops
"plotsList" : "" # 'originalPlot'

} #end of settingsSet dictionary
## ----------------------- End settings for analysis -----------------------


#----------------------Define preprocessing functions----------------------

#--------- TempC2K - convert temperature to Kelvin ---------
def TempC2K():
#Requires a temperature column
    #Save any changes we make to X
    global X
    print('Running preprocessing function: TempC2K')
    #Add in our conversion
    X['temperature'] += 273.15
    #Send our results back to the main path
    return X

#--------- HumidRel2Abs - convert relative humidity to absolute ---------
def humidrel2abs():
#Requires "temperature" and "humidity" columns
#If we get a new pressure sensor, update this to calculate based on pressure
    print('Running preprocessing function: humidrel2abs')
    #Save any changes we make to X
    global X, settingsSet
    #Find humidity & temperature columns
    temp = X['temperature']
    rh = X['humidity']
    
    #Import in toolbox to calculate absolute humidity
    from atmos import calculate
    #Calculates partial pressure of water.  Assumes average sea level atmospheric pressure.
    #(Can change pressure here if necessary)
    #Note: output values here are VERY different from matlab
    temparray= calculate('AH', RH=rh, p=1e5, T=temp, debug=True)
    #get the absolute humidity output value alone
    absHumtemp=temparray[0]
    #Overwrite the humidity column
    X['humidity'] = absHumtemp

    #Send our results back to the main path
    return X

#--------- removeNaNs - remove any missing values ---------
def removeNaNs():
    print('Running preprocessing function: removeNaNs')
    #Save any changes we make to X
    global X
    #delete any empty or nan rows
    X.dropna()

    #Send our results back to the main path
    return X

#--------- podTimeZone - remove any missing values ---------
def podTimeZone():
    print('Running preprocessing function: podTimeZone')
    #Load in the variables we need
    global X, settingsSet, currentPod, n
    #get the deploy log on its own
    deployLog = settingsSet['Deployment Log']
    #get just the columns pertinent to the current pod
    temp = deployLog.loc[currentPod]
    #If more than 1 entry, use n to find the correct value
    if n < 2: #colo or colo2 - use first entry by default
        podTZ = temp.iloc[0,0]
    else: #field data
        podTZ = temp.iloc[1,0]
            
    #Create temporary array to hold times
    import pandas as pd
    X.index = X.index + pd.Timedelta(podTZ, unit='H') 
    
    #Send our results back to the main path
    return X

#--------- makeTElapsed - calculates elapsed time ignoring gaps ---------
def makeTElapsed():
    print('Running preprocessing function: makeTElapsed')
    #Load in the variables we need
    global X, settingsSet
    #get our times in oython's number format
    newt = X.index.values.astype(float)
    #get the elapsed time
    X['telapsed'] = newt - newt[0]
    
    #Send our results back to the main path
    return X

#--------- rmWarmup - removes 1 hr after gaps longer that 5* the typical
# interval between measurements ---------
def rmWarmup():
    print('Running preprocessing function: rmWarmup')
    #Load in the variables we need
    global X, settingsSet, t
    #Length of time (in seconds) to remove after gaps:
    trem = datetime.timedelta(0,settingsSet['timeAvg']*60)
    #Initialize blank dataframe for our calculations
    import pandas as pd
    df=pd.DataFrame()
    #get our time column alone
    df['time']=X.index
    #get the deltas between each measurement
    df['delta'] = df['time'].diff().shift(-1)
    #get the typical delta between measurements
    typdelta = st.median(df['delta'])
    #figure out how many entries to remove
    num_entries=trem/typdelta
    #get rid of sections where gap is 5*delta t or more
    for u in range(len(X)):
        #if the gap is more than 5 * typical delta
        if df.iloc[0,u] > 5*typdelta:
            #remove approx. an hour worth of data
            X.drop[u:u+num_entries-1]
   
    #Send our results back to the main path
    return X

#--------- podSmooth - averages the input timeseries over some interval
def podSmooth():
    print('Running preprocessing function: podSmooth')
    global X, settingsSet, t
    #make another version of X to edit
    X_temp = pd.DataFrame()
    X_temp = X_temp.append(X)
    #get our time column alone
    X_temp['time']=X_temp.index
    #get time averaging for analysis
    tav = settingsSet['timeAvg']
    #get the deltas between each measurement
    X_temp['delta'] = X_temp['time'].diff().shift(-1)
    #get the median delta
    dt = st.median(X_temp['delta'])
    
    #If the data has already been averaged we can skip this
    if dt == tav:
        print('Data already at smoothing interval - smoothing skipped!')
    else:
        #re-average our times
        X = X.resample('{}T'.format(tav)).median()
        
    #smooth each column of X using a 3rd order polynomial
        for kk in range(len(X.iloc[0])):
            if X.columns[kk] != 'telapsed': #don't smooth elapsed time
                X.iloc[:,kk] = savgol_filter(X.iloc[:,kk], 51, 3)
    #Send our results back to the main path
    return X

#------------------------- remove 999 -------------------------
def remove999():
    print('Running preprocessing function: remove999')
    #Save any changes we make to Y
    global Y_ref
    #delete any values containing -999
    Y_ref.replace(-999,'NaN')
 
    #Send our results back to the main path
    return Y_ref
 
#------------------------ remove zeros ------------------------
def removeZeros():
    print('Running preprocessing function: removeZeros')
    #Save any changes we make to Y
    global Y_ref
    #delete any values of 0
    Y_ref.replace(0,'NaN')
 
    #Send our results back to the main path
    return Y_ref
 
#--------- refSmooth - averages the input timeseries over some interval
def refSmooth():
    print('Running preprocessing function: refSmooth')
    global Y_ref
    #make another version of X to edit
    Y_ref_temp = pd.DataFrame()
    Y_ref_temp = Y_ref_temp.append(X)
    #get our time column alone
    Y_ref_temp['time']=Y_ref_temp.index
    #get time averaging for analysis
    tav = settingsSet['timeAvg']
    #get the deltas between each measurement
    Y_ref_temp['delta'] = Y_ref_temp['time'].diff().shift(-1)
    #get the median delta
    dt = st.median(Y_ref_temp['delta'])
  
    #If the data has already been averaged we can skip this
    if dt == tav:
         print('Data already at smoothing interval - smoothing skipped!')
    else:
        #re-average our times
        Y_ref = Y_ref.resample('{}T'.format(tav)).median()
      
    #smooth each column of X using a 3rd order polynomial
    for kk in range(len(Y_ref.iloc[0])):
        Y_ref.iloc[:,kk] = savgol_filter(Y_ref.iloc[:,kk], 51, 3)
             
    #Send our results back to the main path
    return Y_ref   

#----------------------Define validation functions----------------------

#--------- randVal - splits into k-folds randomly ---------
def randVal():
    print('Splitting into validation by: randVal')
    #Save any changes we make to X
    global X, Y_ref, settingsSet, nReps
    #Import in sklearn toolbox
    from sklearn.model_selection import KFold
    #initialize kfolds
    kf = KFold(n_splits=nReps)
    #need to make X an array instead of a dataframe
    X_array = X.to_numpy()
    #get the folds
    kf.get_n_splits(X_array)
    #assign test & train datasets
    for train_index, test_index in kf.split(X):
        X_train, X_test = X_array[train_index], X_array[test_index]
        y_train, y_test = Y_ref[train_index], Y_ref[test_index]
    
    #Send our results back to the main path
    return X_train, X_test, y_train, y_test

#----------------------Define model functions----------------------

#--------- ratioSens1Te - 2600/2602 and elapsed time ---------

def ratioSens1Te():
    #We need to figure out which version we need first - fitting, applying, etc.
    global version
    
    #Define each case first
    
    def ratioSens1TeGen():
        
        #Load in variables we will need to load & save out
        global X, Y_ref, t, Y_hat, rmse, fittedMdls, version, settingsSet
         
        #If we don't have the sensors we need, throw an error
        if settingsSet['podSensors'][0] in X == 0 or settingsSet['podSensors'][1] in X == 0:
            input('Required columns for calibration model not found. cannot proceed with calibration model')
        
        #Otherwise can begin fitting our model
        
        #First column of Y is fitted pollutant
        pollutant = Y_ref.name
        #Make sure we keep only the required columns in case others are present
        X_temp = X.filter(['temperature','humidity','telapsed']
        #Join data into a temporary table
        #C = Y_ref.to_frame()
        
        #return our final estimates
        return fittedMdls, Y_hat
        
    def ratioSens1TeApply():

        return "February"

    def ratioSens1TeApplyColo():
        
        return "March"

    def ratioSens1TeApplyField():
        
        return "April"

    #Create a dictionary object to act as our "switcher"

    switcher = {
        0: 'ratioSens1TeGen',

        1: 'ratioSens1TeApply',

        2: 'ratioSens1TeApplyColo',

        3: 'ratioSens1TeApplyField'
        
    }
            
          
   

## Begin body of code

#Prompt user to select folder for analysis
path = askdirectory(title='Select Folder for analysis').replace("/","\\")

#Add the additional matlab functions as another directory
addt_path = os.path.join(path, 'Addtl Python Functions')
    
# Make sure z-scoring is on for universal calibration
if settingsSet.get("calMode") == 1 and any(["zscoreSensors" in settingsSet]):
    print('Warning: For universal calibration, z-scoring of sensors is required. Please add the correct preprocessing function and try again, or press any key to override.')
    time.sleep()
    
def getFilesList():   
    #Create directories matching each of our folders
    coloPodDir = os.path.join(path, 'Colocation\Pods')
    coloRefDir = os.path.join(path, 'Colocation\Reference')
    colo2PodDir = os.path.join(path, 'Colocation2\Pods')
    fieldPodDir = os.path.join(path, 'Field')
    
    #create file list data
    fileList_data = {'Names': ['colocation pods','colocation reference', 'colocation 2 pods', 'field'], 
                     'Directories': [coloPodDir, coloRefDir, colo2PodDir, fieldPodDir]
                     }
    #combine the data that will become the dataframw
    fileList = pd.DataFrame(fileList_data)
    #add the file names from each folder to the dataframe
    fileList['Files'] = [os.listdir(coloPodDir), os.listdir(coloRefDir), os.listdir(colo2PodDir), os.listdir(fieldPodDir)]
    
    ######get name of reference pollutant????
    #do this how we're doing the pod names maybe??
    
    #add another column to add the pod names to
    fileList['Pod Names']=np.nan
    
    #get the pod name for colo, colo2, & field files
    for i in range(4): #for each column in dataframe
        if i != 1: #excluding the reference column
            #create a list to add new values to
            newtemp=list()
            #get the values we're about to loop through separated
            temp=fileList.iat[i,2]
            for k in range(len(temp)): #loop through the files in each
                #add the YPOD__ string to the new variable
                newtemp.append(temp[k].split("_")[0])
            #add the new strings back into the original dataframe
            fileList.iat[i,3] = newtemp
           #add 
    #get all of the pod names in their own array
    podList=fileList['Pod Names'].tolist()
    #only keep the pod names & combine all into one list
    podList = podList[0] + podList[2] + podList[3]
    #only keep unique pod names
    podList=np.unique(podList)
    
    #Add the 2 new variables to the settingsSet
    settingsSet['fileList'] = fileList
    settingsSet['podList'] = podList
getFilesList()
    
    
# If universal cal, reorder files so calibrating pod is first
if settingsSet.get("calMode") > 0:
    #get the podList out of the settingsSet
    temp=settingsSet['podList'].tolist()
    #get the index of the universal calibrating pod
    if temp.index(settingsSet['uniPod']) != 0:
        #get the name of the first pod
        first = temp[0]
        #get the location of the universal pod
        loc_uni = temp.index(settingsSet['uniPod'])
        #switch their positions
        temp[0] = settingsSet['uniPod']
        temp[loc_uni] = first
        #replace the podList in settingsSet
        settingsSet['podList'] = temp
        #clear variables created here
        del temp, first, loc_uni
        
######Getting a new folder for outputs doesn't work (yet)#############
######################################################################
######################################################################
# def createFolder(directory):
#     try:
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#     except OSError:
#         print ('Error: Creating directory. ' +  directory)

# #Name a new folder to save outputs in
# outpath = os.path.join(path, 'Outputs_{}'.format(datetime.now()))
        
# createFolder(outpath)
######################################################################
######################################################################
 
#save the initial settingsSet as a text file   
with open('settingsSet.txt', 'w') as f: 
    for key, value in settingsSet.items(): 
        f.write('%s:%s\n' % (key, value))

    
#load in files from the logs
logs = ["Deployment Log","Pod Inventory"]

for i in range(len(logs)): #loop through for each of the logs
    logpath = os.path.join(path, 'Logs\{}.xlsx'.format(logs[i]))
    logfile = pd.read_excel(logpath, index_col=0)
    settingsSet["{}".format(logs[i])] = logfile
del logpath, logfile, logs, i, f
    
#The number of reference files, pods, regressions, validations, and folds to evaluate
fileList=settingsSet['fileList']
nref = len(fileList.iat[1,2])
nPods = len(settingsSet['podList'])
nReps = settingsSet['nFoldRep']
nStats = len(settingsSet['statsList'])
nPlots = len(settingsSet['plotsList'])

#---------------------Fit calibration equations---------------------#

#---------------------------Start pod loop---------------------------#
for j in range(nPods):
    #Separate the pod list from the settingsSet
    podList=settingsSet['podList']
    #Get current pod name for readability
    currentPod = podList[j]
        
    #For all individual cal & calibrating pod of universal cal
    if settingsSet['calMode'] == 0 or settingsSet['calMode'] != 0 and settingsSet['uniPod'] == currentPod:
        #Keep track of the loop number in case it's needed by a sub function
        settingsSet['loops'] = j
        print('--- Fitting models for pod {} ----'.format(currentPod))
        
        #Load Pod Data
        print('---- Loading data for {} ----'.format(currentPod))
        
        #define n for colocation data when loading pod data
        n = 0 #colocation
        
        
        def loadPodData():
            ##Loop through list of pod files and upload ones that match the pod name
            #First, get the number of files in each
            nPodFiles = len(fileList.iat[n,2]);
            #Get the directory (folder) where colocated pod data are stored
            fileDir = fileList.iat[n,1];
            
            #Loop through each file in the pod file directory and check if it contains data for this pod
            for i in range(nPodFiles):
                #get the pod name
                filePodName = podList[n]
                #get the list of filenames separate
                temp=fileList.iat[n,2]
                #get the name for this pod file
                fileName = temp[n]
                #delete the temporary variable
                del temp
            
            #Create full file path for reading file
            filePath = os.path.join(fileDir, fileName)
            
            #Load in the data from text file
            temp = pd.read_csv(filePath, sep=',')
            
            #If first loaded file, create the podData variable and assign "temp" to it
            if 'podData' not in locals():
                global podData
                podData = temp
            else:
                podData = np.append(podData,temp)  
        loadPodData()

        #Extract just columns needed
        print('---- Extracting important variables from pod data ----')
        
        def dataExtract():
            #Get the pod inventory alone
            temp = settingsSet['Pod Inventory']
            #Get all values of the current pod in the pod inventory
            tempp = temp[temp["PodName"] == currentPod]
            #Split into comma-separated columns
            headings = tempp['VarNames'].str.split(',', expand=True)
            #rename the original dataframe with the new column headings
            temp_podData = podData.set_axis(headings.iloc[0], axis=1, inplace=False)
            #add the datetime as its own column
            temp_podData['datetime'] = pd.to_datetime(temp_podData['Date'] + ' ' + temp_podData['Time'])
            #get the list of columns to keep
            keep = settingsSet['podSensors'] + settingsSet['envSensors']
            #make our new variables global
            global X, t
            #Now keep only the columns that we need
            X = temp_podData[keep]
            #Also keep the timestamps
            t = temp_podData['datetime']
            X.insert(0,'datetime',temp_podData['datetime'])
            #make the datetime column the index
            X = X.set_index('datetime')
        dataExtract()
        
        #---------------------------Pre-process pod data---------------------------#
        print('Applying selected pod data preprocessing')
        
        #get the names of the preprocessing functions to call
        preProcess = settingsSet['podPreProcess']
        #execue for each of the selected preprocessing functions
        for jj in range(0,len(preProcess)):
            #import the file with the preprocessing functions in it
            #from preprocessingfuns import *
            #call the preprocessing function
            globals()[preProcess[jj]]()
            
        #----------------------Start reference file loop----------------------#
        for i in range(nref):
            #Create empty structures to store fitted models & stats
                #fittedMdls = cell(nModels,nValidation,nReps)
                #mdlStats = cell(nModels,nValidation,nStats)
                #Y_hat.cal = cell(nModels,nValidation,nReps)
                #Y_hat.val = cell(nModels,nValidation,nReps)
                #valList = cell(nValidation,1)
                
            #------------------------Get Reference Data------------------------
            #get the fileList out of the settingsSet
            tempdata = settingsSet['fileList']
            reffilelist = tempdata.iloc[1,2]
            refdirlist = tempdata.iloc[1,1]
            #loop through in case there's more than 1
            for ll in range(len(reffilelist)): 
                currentRef = reffilelist[ll]
                currentfilePath = os.path.join(refdirlist, currentRef)
                
                #Load the reference file into memory
                print('Importing reference file {}'.format(currentRef))
                #Load in the data from text file
                Y_ref = pd.read_csv(currentfilePath)
                #make sure our datetime column is recognized as a pd datetime
                Y_ref['datetime'] = pd.to_datetime(Y_ref['datetime'])
                #make the datetime column the index
                Y_ref = Y_ref.set_index('datetime')
                
                #keep only the columns we need
                if settingsSet['refGas'] not in Y_ref:
                    print('Target gas not found in reference: {}. Combination skipped!'.format(currentRef))
                
                #-------------Pre-process reference data -----------
                #get the names of the preprocessing functions to call
                preProcess = settingsSet['refPreProcess']
                #execue for each of the selected preprocessing functions
                for jj in range(0,len(preProcess)):
                    #import the file with the preprocessing functions in it
                    #from preprocessingfuns import *
                    #call the preprocessing function
                    globals()[preProcess[jj]]()
                    
                #------------- Align ref & pod -----------
                #keep just the pod dates that have matching ref dates
                def alignRefandPod():
                    print('Joining pod & reference data')
                    #make necessary variables global
                    global X, Y_ref, t, fitdata
                    #combine
                    fitdata = pd.merge(X,Y_ref,on='datetime')
                    #save out the timestamps
                    t = fitdata.index
                    #Save out remaining Y values separately
                    Y_ref=fitdata['{}'.format(settingsSet['refGas'])]
                    #delete Y from X
                    del fitdata[settingsSet['refGas']]
                    #save fitdata as new X
                    X = fitdata
                    #Send our results back to the main path
                    return X, Y_ref, t
                alignRefandPod()
                
                #------------- Deploy log match -----------
                #Verify that data in Y and X match time ranges in the deployment log
                def refdeployLogMatch():
                    print('Checking data against the deployment log')
                    #make necessary variables global
                    global currentPod, X, Y_ref, t
                    #get the deploy log out of the settingsSet
                    deployLog = settingsSet['Deployment Log']
                    #loop through entries in the deployment log
                    for mm in range(len(deployLog)):
                        #do for all entries of the current pod only
                        #only keep pod data matching those timestamps
                        if deployLog.index[mm] == currentPod and deployLog.iloc[mm,3] == currentRef:
                            #pull our start & end dates from the deploy log
                            startdate = deployLog.iloc[mm,1]
                            enddate = deployLog.iloc[mm,2]
                            #create the "mask" to pull between these dates
                            mask = (X.index > startdate) & (X.index <= enddate)
                            #apply the "mask" to X
                            X = X.loc[mask]
                            #correct t
                            t = X.index
                        
                            #create the "mask" to pull between these dates
                            refmask = (Y_ref.index > startdate) & (Y_ref.index <= enddate)
                            #apply the "mask" to Y
                            Y_ref = Y_ref.loc[refmask]
                    #Send our results back to the main path
                    return X, Y_ref, t
                refdeployLogMatch()    
                
                #Skip this run if there is no overlap between data and entries in deployment log
                if len(t) == 0:
                    print('No overlap between data and entries in deployment log for {} and {}. This combo will be skipped!'.format(currentPod,currentRef))
                    del X, Y_ref, t
                
                #----------------- START VALIDATION SETS LOOP -----------------
                #Get the validation function on its own
                validFunc = settingsSet['valList']
                
                #if there's only 1, call just the 1
                if isinstance(validFunc, str):
                    #execute the validation function
                    globals()[validFunc]()
                #if there's more than 1, need to loop through each
                elif isinstance(validFunc,list):
                    #loop through and run each function
                    for zz in range(1,len(validFunc)):
                        #import the file with the preprocessing functions in it
                        #from preprocessingfuns import *
                        #call the preprocessing function
                        globals()[validFunc[zz]]()
                    
                #------------------START REGRESSIONS LOOP----------------------
                #get the list of models to run 
                models = settingsSet['modelList']
                
                #if there's only 1, call just the 1
                if isinstance(models, str):
                    #create an array to hold estimates from each rep
                    column_names = ["cal", "val", "fold"]
                    Y_hat = pd.DataFrame(columns = column_names)
                    #Get an array ready for comparing RMSE's of models
                    rmse = pd.DataFrame(np.zeros((nReps, 1)))
                    #Get an array to hold the models for each fold
                    fittedMdls = pd.DataFrame(index=range(1,nReps))
                    #run the fitting version of the model (v1)
                    version = 1
                    globals()[models]()
                    
                    #run just this one function
                    for kk in range(0,nReps):
                        print('Using calibration/validation fold {}'.format(kk));
                        #execute the model function
                        globals()[models]()
                #if there's more than 1, need to loop through each
                elif isinstance(models,list):
                    #loop through and run each function
                    for uu in range(0,len(models)):
                        #repeat process for each number of reps
                        for kk in range(1,nReps):
                            #execute the model function
                            globals()[models[uu]]()
                        
          