# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 20:32:19 2024

@author: Criss
"""

#import os
#os.environ["OMP_NUM_THREADS"] = '1' # this is necessary to avoid memory leak

#os.chdir('C:\\Users\\Criss\\Documents\\Lavoro\\Assegno_2024_2025\\Codici')

# Fundamental libreries
import numpy as np
import pandas as pd
import glob
import xarray as xr

from functions_new_version import *
from SOM_variable_file import *

# PATHS
path_extremes="C:/Users/Criss/Documents/Lavoro/Assegno_2024_2025/Dati/" # where you have saved the csv files
output_path ="C:/Users/Criss/Documents/Lavoro/Assegno_2024_2025/Codici/SOM/new_extremes/"

#State the path where the file is located. This will be the same path used in MiniSOM Tutorial Step #1
PATH ="C:/Users/Criss/Documents/Lavoro/Assegno_2024_2025/Codici/SOM/new_extremes/" #This is the path where the data files
folderpath = 'C:/Users/Criss/Documents/Lavoro/Assegno_2024_2025/Codici/SOM/SOMs_output/' 

# datasets
data_path_ERA5 = "C:/Users/Criss/Documents/Lavoro/Assegno_2024_2025/Dati/ERA5/" 
#path_arcis = "C:/Users/Criss/Documents/Lavoro/Assegno_2024_2025/Dati/ArCIS"
#path_mswep = "C:/Users/Criss/Documents/Lavoro/Assegno_2024_2025/Dati/MSWEP"
#path_cerra = "C:/Users/Criss/Documents/Lavoro/Assegno_2024_2025/Dati/CERRA_LAND"

#%% Data and paths: ERA5
data_path_ERA5 = "C:/Users/Criss/Documents/Lavoro/Assegno_2024_2025/Dati/ERA5/" # path ERA5

### Z500 and MSLP data
#Z500
name_file = data_path_ERA5 + "ERA5_daily_mean_Geop_500hPa_1985_2019_" + EU_domain + ".nc" 
ds = xr.open_mfdataset(name_file, preprocess=select_latlon)
#ds = dy.sel(time=extreme_list, method="nearest") 
#print(ds)

# Loading the data into the variables
z_time_values = ds['time'].values
z_values = ds['Z'].values
z_values = z_values / g
z_values = z_values[:,0,:,:]
z_raw = (ds['Z'])/g  #This is the data the NON-anomaly data.
z_lon = ds['lon'].values
z_lat = ds['lat'].values

# MSLP
name_file = data_path_ERA5 + "ERA5_daily_mean_mSLP_sfc_1985_2019_" + EU_domain + ".nc" 
ds = xr.open_mfdataset(name_file, preprocess=select_latlon)
print(ds)

# Loading the data into the variables
mslp_time_values = ds['time'].values
mslp_values = ds['MSL'].values
mslp_values = mslp_values /100
mslp_raw = (ds['MSL'])/100 #This is the data the NON-anomaly data.


## IVT data
name_file = "ERA5_vertical_integral_of_eastward_water_vapour_flux_day_full_sfc_EU_1985_2022.nc"

path_file = data_path_ERA5 + name_file

viwve, time_ERA5, xlat_ERA5, xlong_ERA5, _ = ERA5_IVT(path_file, EU_domain, "time2")
viwve = viwve[:,0,:,:]

name_file = "ERA5_vertical_integral_of_northward_water_vapour_flux_day_full_sfc_EU_1985_2022.nc"

path_file = data_path_ERA5 + name_file

viwvn, _, _, _, _ = ERA5_IVT(path_file, EU_domain, "time2")
viwvn = viwvn[:,0,:,:]

## TCWV
name_file = data_path_ERA5 + "ERA5_total_column_water_vapour_day_full_sfc_EU_1985_2022.nc" 
ds = xr.open_mfdataset(name_file, preprocess=select_latlon)
print(ds)

# Loading the data into the variables
TCWV_time_values = ds['time'].values
TCWV_values = ds['TCWV'].values
TCWV_raw = (ds['TCWV']) #This is the data the NON-anomaly data.
lon = ds['lon'].values
lat = ds['lat'].values
nx = int((TCWV_raw['lat'].size))
ny = int((TCWV_raw['lon'].size))

#%% Variables and paths: precipitation
output_path ="C:/Users/Criss/Documents/Lavoro/Assegno_2024_2025/Codici/SOM/new_extremes/"

extreme_list_str = "CERRA_LAND"
#extreme_list = np.load(output_path + extreme_list_str + '_extreme_dates.npy', allow_pickle=True)
extreme_list = np.load(output_path + extreme_list_str + '_extreme_dates_Italy.npy', allow_pickle=True)

# carico i dati
path_CERRA = "C:/Users/Criss/Documents/Lavoro/Assegno_2024_2025/Dati/CERRA_LAND"            # path 
#pr_file = path_MSWEP + "/MSWEP_1991_2020_Italy.nc"

pr_file = path_CERRA + "/All_precip_CERRA_LAND_1984_2021_italy.nc"

dy = xr.open_mfdataset(pr_file)
ds = dy.sel(time=extreme_list, method="nearest") 
pr_time_CERRA = ds['time'].values
#pr_time_MSWEP = ds['time'].dt.date.values
#pr_data = ds['precipitation'].values
pr_raw_CERRA = ds['tp']
pr_CERRA = ds['tp'].values

pr_lon = ds['lon']
pr_lat = ds['lat']
pr_nx = int((pr_raw_CERRA['lat'].size))
pr_ny = int((pr_raw_CERRA['lon'].size))

#%%
mask_file = path_CERRA + "/prova_mask.nc"
dy = xr.open_mfdataset(mask_file)
print(dy)

mask_lon = dy['lon']
mask_lat = dy['lat']
mask_nx = int((mask_lat.size))
mask_ny = int((mask_lon.size))

print(pr_lon, mask_lon)
print(pr_lat, mask_lat)
#%%
# ok sono stessa dimensione ora
mask = dy["stl1"].values
sea_mask = mask/mask

#%%
pr_mask = pr_CERRA * sea_mask
#%%
mask = dy["stl1"].values
prova_mask = mask/mask
prova_mask[0,:60,:] = np.nan #lat
prova_mask[0,180:,:] = np.nan 
prova_mask[0,:,:50] = np.nan #lon
prova_mask[0,:,190:] = np.nan
