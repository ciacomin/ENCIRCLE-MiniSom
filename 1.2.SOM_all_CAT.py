# Following this tutorial
# https://github.com/taliakurtz/MiniSOM_tutorial/blob/main/MiniSOM_Tutorial_Step_1.ipynb

import os

os.chdir('C:\\Users\\Criss\\Documents\\Lavoro\\Assegno 2024_2025\\Codici')

#Imports
import xarray as xr
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from functions_new_version import *

#%matplotlib inline

data_path = "C:/Users/Criss/Documents/Lavoro/Assegno 2024_2025/Dati/ERA5/"

output_path ="C:/Users/Criss/Documents/Lavoro/Assegno 2024_2025/Codici/SOM/"

#Functions
def select_latlon(ds):
    return ds.sel(lat = slice(60,36), lon = slice(-10,19)) #change to your lat/lon

#34N - 48M   4E - 16E

#%% Prepare the dataset for the extracting of the extreme dates
path_extremes="C:/Users/Criss/Documents/Lavoro/Assegno 2024_2025/Extreme_pr_Grazzini"

# i need to read two csv files
dates_file=  path_extremes + "/EPE_NItaly_Cat_1991-2022.csv"

dates_raw= dataframe_to_datatime(dates_file)

# Getting the "CAT" days
extreme_dates = dates_raw["Time"]
extreme_list = extreme_dates.tolist()

#%% Z500
name_file = "ERA5_daily_mean_Geop_500hPa_1991_2022_EU.nc"

dy = xr.open_mfdataset(data_path + name_file,    #Xarray features will be used throughout this tutorial 
                         preprocess=select_latlon)
ds = dy.sel(time=extreme_list, method="nearest") 
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
z_arr_mean = np.empty(nday) #This is the mean array 

#We are now going to place the raw Z data into the array (z_arr)
for i in range(nday):
    z_arr[i,:]= z_raw[i,:,:].stack(point=["lat", "lon"])
     
#We are now calculating the hourly anomaly data. The hourly mean will be removed from the data. 
for i in range(nday):
    z_arr[i,:] =z_arr[i,:]-np.mean(z_arr[i,:])
    z_arr_mean[i] = np.mean(z_arr[i,:])


max_value=-9999999
min_value=999999   #we are setting the min value and max value variables to a value so that there is no junk in the variable and each will easily overcome the set value.

for i in range(nday):
    min_value=min(min_value,np.min(z_arr[i,:]))
    max_value=max(max_value,np.max(z_arr[i,:]))
print("max: ", max_value, "; min: ", min_value)

#We are generating the MSLP factor to be multipled to the data to normalize it
z500_factor=100./(max_value-min_value)
print("All CAT: " + str(z500_factor))

#The data is now being normalized.
data_train = z_arr*z500_factor
#print
#data_train = z_arr
data_train.shape

z_raw.to_netcdf(output_path + 'Z500_som_raw_All_CAT_anomalies.nc')
np.save(output_path + 'Z500_som_data_train_All_CAT_anomalies.npy', data_train)
np.save(output_path + 'Z500_som_time_data__All_CAT_anomalies.npy', time_values)
np.save(output_path + 'Z500_mean_array_All_CAT_anomalies.npy', z_arr_mean)


#%% MSLP
name_file = "ERA5_daily_mean_mSLP_sfc_1991_2022_EU.nc"

dy = xr.open_mfdataset(data_path + name_file,    #Xarray features will be used throughout this tutorial 
                         preprocess=select_latlon)
ds = dy.sel(time=extreme_list, method="nearest") 
print(ds)

# Loading the data into the variables
time_values = ds['time'].values
mslp_values = ds['MSL'].values
mslp_values = mslp_values /100
mslp_raw = (ds['MSL'])/100 #This is the data the NON-anomaly data.
lon = ds['lon'].values
lat = ds['lat'].values

#generate the empty array that will house the data.
nday =int((ds['time'].size))
nlat = int((ds['lat'].size))
nlon = int((ds['lon'].size))
mslp_arr = np.empty((nday, nlat*nlon))  #This is the new array that we will place the data into. 
mslp_arr_mean = np.empty(nday) # this is the mean array

#We are now going to place the raw Z data into the array (z_arr)
for i in range(nday):
    mslp_arr[i,:]= mslp_raw[i,:,:].stack(point=["lat", "lon"])
     
#We are now calculating the hourly anomaly data. The hourly mean will be removed from the data. 
for i in range(nday):
    mslp_arr[i,:] =mslp_arr[i,:]-np.mean(mslp_arr[i,:])
    mslp_arr_mean[i] = np.mean(mslp_arr[i,:])


max_value=-9999999
min_value=999999   #we are setting the min value and max value variables to a value so that there is no junk in the variable and each will easily overcome the set value.

for i in range(nday):
    min_value=min(min_value,np.min(mslp_arr[i,:]))
    max_value=max(max_value,np.max(mslp_arr[i,:]))
print("max: ", max_value, "; min: ", min_value)

#We are generating the MSLP factor to be multipled to the data to normalize it
mslp_factor=100./(max_value-min_value)
print("All CAT: " + str(mslp_factor))

#The data is now being normalized.
data_train = mslp_arr*mslp_factor

#data_train = z_arr
data_train.shape

mslp_raw.to_netcdf(output_path + 'MSLP_som_raw_All_CAT_anomalies.nc')
np.save(output_path + 'MSLP_som_data_train_All_CAT_anomalies.npy', data_train)
np.save(output_path + 'MSLP_som_time_data__All_CAT_anomalies.npy', time_values)
np.save(output_path + 'MSLP_mean_array_All_CAT_anomalies.npy', mslp_arr_mean)
