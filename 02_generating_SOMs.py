"""
@author: ciacomin
"""

import os
# Fundamental libreries
import numpy as np
import pandas as pd
import glob
from enum import Enum
import xarray as xr
import winsound

# figures
import geopandas as gpd
import cartopy.crs as ccrs
crs = ccrs.PlateCarree()
import cartopy.feature as cfeature
from matplotlib import ticker

os.chdir('C:\\your_directory') # if you need to change to your directory 

from function_SOMs import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from itertools import product

#import for the SOM
import minisom
from minisom import asymptotic_decay
from datetime import date
import pickle

from sklearn.metrics import silhouette_score

def getList(dict):
    list = []
    for key in dict.keys():
        list.append(key)
        
    return list

class RegionDomain(Enum):
    NORTH_ITALY = "north"
    ITALY = "italy"
    
class Dataset(Enum):
    CERRA_LAND = "cerra-land6"
    ARCIS = "arcis3"
    MSWEP = "mswep"
    
class DatasetString(Enum):
    CERRA_LAND = "CERRA_LAND"
    ARCIS = "ARCIS"
    MSWEP = "MSWEP"
    
def select_latlon(ds):
    # this function selects the domain for the SOMs
    
    return ds.sel(lat = slice(52,27), lon = slice(-13,35)) # 

# PATHS
path_extremes="data/" # where you have saved the csv files of the extreme
output_path ="output_SOM/" # the output path
#folderpath_tuning = 'tuningFinale/'  # Output of tuning section, if you want to save your results
folderpath_SOMs = 'output_SOM/SOMs/'

# datasets
data_path_ERA5 = "data/ERA5/" 
################################################################


## MY FINAL CONFIGURATION
# gaussian
# 1985 - 2019
# ARCIS, 2x2: sigma = 1, LR = 0.002
# CERRA Land, 3x2: sigma = 1, LR = 0.005

# 1984 - 2021
# CERRA Land, 3x2: sigma = 0.95, LR = 0.0008

#%% 0) Dataset and variable selection 
# Chose first which dataset and domain region
dataset = Dataset.CERRA_LAND.value                      # chose the desired dataset
pr_dataset_str = DatasetString.CERRA_LAND.value         # same as before 
region_domain = RegionDomain.ITALY.value                # chose the desired region domain
# Chose the domain for the SOM
SOM_domain = "EU5" #SOM domain

start_year = 1984
end_year = 2021

#%% 1) Load the extreme dates 

print("I'm loading the extremes of " + pr_dataset_str + '_extreme_dates_' + region_domain + "_" + str(start_year) + "_" + str(end_year) + '.npy')
extreme_pr_days_list = np.load(output_path + pr_dataset_str + '_extreme_dates_' + region_domain + "_" + str(start_year) + "_" + str(end_year) + '.npy', allow_pickle=True)
print(extreme_pr_days_list.shape)

#%% 2) Preparing the SOM - Z500 and mSLP

# Here you choose the domain for the SOM and preparing the files you need
# 2a) load the variable of interest and do the standardization (per each variable)
# 2b) start the pre-processing with the data train - test split
# 2c) PCA

print("2) PREPARING THE DATA FOR THE SOM ")
print("   SOM domain: "+ SOM_domain)
#dataset_str = extreme_list_str # CHANGE THIS TO pr_dataset_str

# 2a) load the variable of interest
print(" -Loading the variables-")
print("   Loading Z500...")

name_file_z500 = "ERA5_daily_mean_Geop_500hPa_" + str(start_year) + "_" + str(end_year) + "_EU3_anomalies.nc"           # Anomalies of Z500
dy = xr.open_mfdataset(data_path_ERA5 + name_file_z500, preprocess=select_latlon)   # Here you select the domain
ds = dy.sel(time=extreme_pr_days_list, method="nearest")                            # Here you select the extreme days
#print(ds)

time_values = ds['time'].values
z_values = ds['Z'].values
z_values = z_values / g
lon = ds['lon'].values  #they are the same for the two variables 
lat = ds['lat'].values

nday =int((ds['time'].size))
nlat = int((ds['lat'].size))
nlon = int((ds['lon'].size))

#  reshape and standardization ( new_x = x - x_mean / standard_deviation )
z500_2d = z_values.reshape(nday, -1)        #reshape 
scaler_z500 = StandardScaler()
z_values_scaled = scaler_z500.fit_transform(z500_2d)

# loading the second variable
# MSLP 
print("   Loading mslp...")
name_file_mslp = "ERA5_daily_mean_mSLP_sfc_" + str(start_year) + "_" + str(end_year) + "_EU3_anomalies.nc"
dy = xr.open_mfdataset(data_path_ERA5 + name_file_mslp,    #Xarray features will be used throughout this tutorial 
                         preprocess=select_latlon)
ds = dy.sel(time=extreme_pr_days_list, method="nearest") 
#print(ds)

#time_values = ds['time'].values    #should be the same
mslp_values = ds['MSL'].values
mslp_values = mslp_values /100
som_variables_str = "Z500_mSLP"
#Z500

mslp_2d = mslp_values.reshape(nday, -1)        #reshape 
scaler_mslp = StandardScaler()
mslp_scaled = scaler_mslp.fit_transform(mslp_2d)

# reshape and standardization ( new_x = x - x_mean / standard_deviation )
z500_reshaped = z_values_scaled.reshape(nday, nlat, nlon)
print(z500_reshaped.shape)
mslp_reshaped = mslp_scaled.reshape(nday, nlat, nlon)

print("  check the shape:")
print("   z500_reshaped", z500_reshaped.shape, "   mslp_reshaped", mslp_reshaped.shape)

temp_array = np.stack([z500_reshaped, mslp_reshaped], axis=-1)   #stack into new axis
temp_array.shape
z500_mslp_arr = temp_array.reshape(nday, -1)        #reshape 

print("  stacking Z500 and mSLP,")
print("   new array shape: ", z500_mslp_arr.shape)

# 2b) Pre-processing
print(" -Preprocessing data train and test split-")
# data train - test SPLIT
print("   lenght of all data:" + str(len(time_values)))
train_perc = round(len(time_values)*70/100)
print("   lenght of train data:" + str(train_perc))
test_perc = len(time_values) - train_perc
print("   lenght of test data:" + str(test_perc))

data_train = z500_mslp_arr[:train_perc]
data_test = z500_mslp_arr[train_perc:]
all_data = z500_mslp_arr
print("   data train shape:", data_train.shape)
print("   data test shape: ", data_test.shape)

#%% INITIALIZE THE SOM
# Now that you have the setup you can create your SOMs
# 1985 - 2019
# ARCIS, 2x2: sigma = 1, LR = 0.002
# CERRA Land, 3x2: sigma = 1, LR = 0.005
#######
sigma = 0.95
learning_rate = 0.0008
perc_pca = 0.95
#######

som_col = 3
som_row = 2
som_shape =(som_row, som_col)
min_som = min(som_col, som_row)
number_of_soms = 10
q_win = 100000.
q_error_list = []
t_error_list = []

pca = PCA(n_components= perc_pca) # Keeps 95% of the variance
data_train_pca = pca.fit_transform(data_train)
data_test_pca = pca.transform(data_test)
all_data_pca = pca.transform(all_data)
print(data_train_pca.shape)

som_names = pr_dataset_str + "_" + region_domain + "_" + SOM_domain + "_" + som_variables_str + "_PCA_" + str(perc_pca) + "_" + str(som_col) + "by" + str(som_row) + "_LR" + str(learning_rate) + "_sig" + str(sigma) + "_n_"
print(som_names)

