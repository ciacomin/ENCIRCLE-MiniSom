# Following this tutorial
# https://github.com/taliakurtz/MiniSOM_tutorial/blob/main/MiniSOM_Tutorial_Step_1.ipynb

##### 
# This script generates a data_train file for each CAT category

import os

os.chdir('C:\\Users\\Criss\\Documents\\Lavoro\\Assegno 2024_2025\\Codici')

#Imports
import xarray as xr
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from functions_new_version import *

data_path = "C:/Users/Criss/Documents/Lavoro/Assegno 2024_2025/Dati/ERA5/"   # where you have stored the .nc files
output_path ="C:/Users/Criss/Documents/Lavoro/Assegno 2024_2025/Codici/SOM/" # your output folder

#%% Prepare the dataset for the extracting of the extreme dates
path_extremes="C:/Users/Criss/Documents/Lavoro/Assegno 2024_2025/Extreme_pr_Grazzini"

# I need to read two csv files
dates_file=  path_extremes + "/EPE_NItaly_Cat_1991-2022.csv"

dates_raw= dataframe_to_datatime(dates_file)

# Getting the "CAT" days
extreme_dates = dates_raw["Time"]
extreme_dates_CAT1 = dates_raw[dates_raw["Cat"].isin([1])]["Time"]
extreme_dates_CAT2 = dates_raw[dates_raw["Cat"].isin([2])]["Time"]
extreme_dates_CAT3 = dates_raw[dates_raw["Cat"].isin([3])]["Time"]
extreme_list = extreme_dates.tolist()

extreme_dates_CAT1_list=extreme_dates_CAT1.tolist()
extreme_dates_CAT2_list=extreme_dates_CAT2.tolist()
extreme_dates_CAT3_list=extreme_dates_CAT3.tolist()

extreme_dates_CAT_list = [extreme_dates_CAT1_list, extreme_dates_CAT2_list, extreme_dates_CAT3_list]

#%% Z500
# if you want to study Z500
name_file = "ERA5_daily_mean_Geop_500hPa_1991_2022_EU.nc"

for k in range(len(extreme_dates_CAT_list)):
    dy = xr.open_mfdataset(data_path + name_file,    #Xarray features will be used throughout this tutorial 
                             preprocess=select_latlon)
    ds = dy.sel(time=extreme_dates_CAT_list[k], method="nearest") 
    print(ds)
    
    # Loading the data into the variables
    time_values = ds['time'].values
    z_values = ds['Z'].values
    z_values = z_values / g
    z_raw = (ds['Z'])/g  #This is the data the NON-anomaly data.
    lon = ds['lon'].values
    lat = ds['lat'].values

    #generate the empty array that will house the data.
    nday =int((ds['time'].size))
    nlat = int((ds['lat'].size))
    nlon = int((ds['lon'].size))
    z_arr = np.empty((nday, nlat*nlon))  #This is the new array that we will place the data into. 

    #We are now going to place the raw Z data into the array (z_arr)
    for i in range(nday):
        z_arr[i,:]= z_raw[i,:,:].stack(point=["lat", "lon"])
         
    # If you want to study the anomaly uncomment this
    #for i in range(nday):
    #    z_arr[i,:] =z_arr[i,:]-np.mean(z_arr[i,:])
    
    
    max_value=-9999999
    min_value=999999   #we are setting the min value and max value variables to a value so that there is no junk in the variable and each will easily overcome the set value.

    for i in range(nday):
        min_value=min(min_value,np.min(z_arr[i,:]))
        max_value=max(max_value,np.max(z_arr[i,:]))
    print("max: ", max_value, "; min: ", min_value)

    #We are generating the Z500 factor to be multipled to the data to normalize it
    z500_factor=100./(max_value-min_value)
    print("CAT" + str(k+1) + ": " + str(z500_factor)) # Save it for later

    #The data is now being normalized.
    data_train = z_arr*z500_factor
    #print
    #data_train = z_arr
    data_train.shape
    
    z_raw.to_netcdf(output_path + 'VER2_SOM_Z_raw_CAT' +  str(k+1) + '_norm.nc')
    np.save(output_path + 'TEST2_som_data_train_CAT' +  str(k+1) + '_norm.npy', data_train)
    np.save(output_path + 'TEST2_som_time_data_CAT' +  str(k+1) + '_norm.npy', time_values)

    

#%% MSLP
# if you want to study MSLP
name_file = "ERA5_daily_mean_mSLP_sfc_1991_2022_EU.nc"

for k in range(len(extreme_dates_CAT_list)):
    dy = xr.open_mfdataset(data_path + name_file,    #Xarray features will be used throughout this tutorial 
                             preprocess=select_latlon)
    ds = dy.sel(time=extreme_dates_CAT_list[k], method="nearest") 
    print(ds)
    
    # Loading the data into the variables
    time_values = ds['time'].values
    z_values = ds['MSL'].values
    z_values = z_values /100
    z_raw = (ds['MSL'])/100 #This is the data the NON-anomaly data.
    lon = ds['lon'].values
    lat = ds['lat'].values

    #generate the empty array that will house the data.
    nday =int((ds['time'].size))
    nlat = int((ds['lat'].size))
    nlon = int((ds['lon'].size))
    z_arr = np.empty((nday, nlat*nlon))  #This is the new array that we will place the data into. 

    #We are now going to place the raw Z data into the array (z_arr)
    for i in range(nday):
        z_arr[i,:]= z_raw[i,:,:].stack(point=["lat", "lon"])


    # If you want to study the anomaly uncomment this
    #for i in range(nday):
    #    z_arr[i,:] =z_arr[i,:]-np.mean(z_arr[i,:])
    
    
    max_value=-9999999
    min_value=999999   #we are setting the min value and max value variables to a value so that there is no junk in the variable and each will easily overcome the set value.

    for i in range(nday):
        min_value=min(min_value,np.min(z_arr[i,:]))
        max_value=max(max_value,np.max(z_arr[i,:]))
    print("max: ", max_value, "; min: ", min_value)

    #We are generating the MSLP factor to be multipled to the data to normalize it
    mslp_factor=100./(max_value-min_value)
    print("CAT" + str(k+1) + ": " + str(mslp_factor)) # Save it for later

    #The data is now being normalized.
    data_train = z_arr*mslp_factor
    
    #data_train = z_arr
    data_train.shape
    
    z_raw.to_netcdf(output_path + 'VER2_SOM_MSLP_raw_CAT' +  str(k+1) + '_norm.nc')
    np.save(output_path + 'TEST2_som_MSLP_data_train_CAT' +  str(k+1) + '_norm.npy', data_train)
    np.save(output_path + 'TEST2_som_MSLP_time_data_CAT' +  str(k+1) + '_norm.npy', time_values)
