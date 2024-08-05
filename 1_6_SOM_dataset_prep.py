# -*- coding: utf-8 -*-
"""
Created on Mon May 13 17:16:42 2024

The first sections can be used to compute the extreme list and then save it. 
If you have already done this passage you can just load the respective file

@author: Criss
"""

import os
#os.environ["OMP_NUM_THREADS"] = '1' # this is necessary to avoid memory leak

os.chdir('C:\\Users\\Criss\\Documents\\Lavoro\\Assegno_2024_2025\\Codici')

# Fundamental libreries
import numpy as np
import pandas as pd
#import glob
import xarray as xr
#from xarray import DataArray

from functions_new_version import *

# PATHS
path_extremes="C:/Users/Criss/Documents/Lavoro/Assegno_2024_2025/Dati/" # where you have saved the csv files
output_path ="C:/Users/Criss/Documents/Lavoro/Assegno_2024_2025/Codici/SOM/new_extremes/"

# datasets
data_path_ERA5 = "C:/Users/Criss/Documents/Lavoro/Assegno_2024_2025/Dati/ERA5/" 
#path_arcis = "C:/Users/Criss/Documents/Lavoro/Assegno_2024_2025/Dati/ArCIS"
#path_mswep = "C:/Users/Criss/Documents/Lavoro/Assegno_2024_2025/Dati/MSWEP"
#path_cerra = "C:/Users/Criss/Documents/Lavoro/Assegno_2024_2025/Dati/CERRA_LAND"
################################################################
# 1) Chose first which dataset and compute the extreme_list or load it in the 2) part
#%% ARCIS 3  - North
#dates_file=  path_extremes + "/EPE_NItaly_Cat_1991-2022.csv"   # extreme old dates file
extreme_list_str = "ARCIS"                     
All_pr_file= path_extremes + "arcis3_19610101_20231231_direct_north.xlsx"            # precipitation file


#%% MSWEP - North
extreme_list_str = "MSWEP"
All_pr_file= path_extremes + "mswep_19800101_20191231_direct_north.xlsx"  


#%% CERRA LAND - North
extreme_list_str = "CERRA_LAND"
All_pr_file= path_extremes + "cerra-land_19840801_20210430_direct_north.xlsx"  

#%% 
print("Sto calcolando gli estremi per " + extreme_list_str)

All_pr_raw, All_pr_cut = dataframe_cut_xlsx(All_pr_file, 1985, 2019, True)      # we want 1985 - 2019 period
pr_extreme_arcis = dataframe_extremes_xlsx(All_pr_cut, "intense rain day")

# Getting the extreme days
extreme_dates = pr_extreme_arcis["Time"]
extreme_list = extreme_dates.tolist()

print("Sto salvando in "+ extreme_list_str + '_extreme_dates.npy')
#np.save(output_path + extreme_list_str + '_extreme_dates.npy', np.array(extreme_list, dtype=object), allow_pickle=True)

#%% CERRA LAND - Italy
extreme_list_str = "CERRA_LAND"
All_pr_file= path_extremes + "cerra-land_19840801_20210430_direct_italy.xlsx" 

# domain
domain_region="Italy"
#domain_region="Sicilia"
#domain_region = "Puglia"

print("Sto calcolando gli estremi per " + domain_region)

All_pr_raw, All_pr_cut = dataframe_cut_xlsx_sheet(All_pr_file, domain_region, 1985, 2019, True)  
pr_extreme_cerra = dataframe_extremes_xlsx(All_pr_cut, "intense rain day")

# Getting the extreme days
extreme_dates = pr_extreme_cerra["Time"]
extreme_list = extreme_dates.tolist()
extreme_list_str = "CERRA_LAND"

print("Sto salvando in "+ extreme_list_str + '_extreme_dates_' + domain_region + '.npy')
np.save(output_path + extreme_list_str + '_extreme_dates_' + domain_region + '.npy', np.array(extreme_list, dtype=object), allow_pickle=True)

############################
#%% 2) if you want directly to load the extremes
#extreme_list_str = "ARCIS"  
#extreme_list_str = "MSWEP"
extreme_list_str = "CERRA_LAND"

# domain
domain_region="Italy"
#domain_region="Sicilia"
#domain_region = "Puglia"

print("Sto caricando gli estremi di "+ extreme_list_str + '_extreme_dates_' + domain_region + '.npy')
np.load(output_path + extreme_list_str + '_extreme_dates_' + domain_region + '.npy', allow_pickle=True)


##########################################################
#%% Preparing the SOM - Z500 and mSLP or Z500 and TCWV
# then prepare the files for the SOM
# Choose here the domain
def select_latlon(ds):
    #return ds.sel(lat = slice(60,36), lon = slice(-10,19)) #EU_1
    #return ds.sel(lat = slice(60,35), lon = slice(-15,30))#EU_2
    return ds.sel(lat = slice(60,25), lon = slice(-13,35))#EU_3 (aggiornato)
    #return ds.sel(lat = slice(47,25), lon = slice(-13,35))#EU_3_MEDITERRANEO -> EU4
#34N - 48M   4E - 16E
EU_domain = "EU3"

dataset_str = extreme_list_str

#name_file_z500 = "ERA5_daily_mean_Geop_500hPa_1985_2019_EU2_anomalies.nc" # 2nd climatology
name_file_z500 = "ERA5_daily_mean_Geop_500hPa_1985_2019_" + EU_domain + "_anomalies.nc" # 2nd climatology
dy = xr.open_mfdataset(data_path_ERA5 + name_file_z500,    #Xarray features will be used throughout this tutorial 
                         preprocess=select_latlon)
ds = dy.sel(time=extreme_list, method="nearest") 
#print(ds)

time_values = ds['time'].values
z_values = ds['Z'].values
z_values = z_values / g
z_raw = (ds['Z'])/g  #This is the data the NON-anomaly data.
z_raw = z_raw[:,0,:,:]
lon = ds['lon'].values
lat = ds['lat'].values

#### Visualize the domain
fig = plt.figure(figsize=(12, 8))    
fig.suptitle("Domain", fontsize=20)

# Figure - base
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
im_0 = plt.contourf(lon, lat, z_values[0,0])
ax.add_feature(cfeature.COASTLINE)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False, linewidth=0.33, color='k', alpha=0.5)
gl.right_labels = gl.top_labels = False
#%%
# loading the secnond variable: uncomment the one you are interested into
# MSLP 
name_file_mslp = "ERA5_daily_mean_mSLP_sfc_1985_2019_" + EU_domain + "_anomalies.nc"
dy = xr.open_mfdataset(data_path_ERA5 + name_file_mslp,    #Xarray features will be used throughout this tutorial 
                         preprocess=select_latlon)
ds = dy.sel(time=extreme_list, method="nearest") 
#print(ds)

#time_values = ds['time'].values    #should be the same
mslp_values = ds['MSL'].values
mslp_values = mslp_values /100
mslp_raw = (ds['MSL'])/100 #This is the data the NON-anomaly data.
som_variables_str = "Z500_mSLP"
#Z500
"""
name_file_mslp = "ERA5_total_column_water_vapour_day_full_sfc_EU_1985_2022_anomalies.nc"
dy = xr.open_mfdataset(data_path_ERA5 + name_file_mslp,    #Xarray features will be used throughout this tutorial 
                         preprocess=select_latlon)
ds = dy.sel(time=extreme_list, method="nearest") 
#print(ds)

#time_values = ds['time'].values    #should be the same
mslp_values = ds['TCWV'].values # it's not mslp this time but i do not want to change all variables names
mslp_values = mslp_values 
mslp_raw = (ds['TCWV']) #This is the data the NON-anomaly data.
som_variables_str = "Z500_TCWV"
"""

#%Train test split
print("lenght of all data:" + str(len(time_values)))
train_perc = round(len(time_values)*70/100)
print("lenght of train data:" + str(train_perc))
test_perc = len(time_values) - train_perc
print("lenght of test data:" + str(test_perc))

nday =int((ds['time'].size))
nlat = int((ds['lat'].size))
nlon = int((ds['lon'].size))

#name_file_mslp = "ERA5_daily_mean_mSLP_sfc_1985_2019_EU2_anomalies.nc"

print(z_raw.shape, mslp_raw.shape)

temp_array = np.stack([z_raw, mslp_raw], axis=-1)   #stack into new axis
temp_array.shape
#z500_mslp_arr = np.empty((nday, nlat*nlon*2))      #This is the new array that we will place the data into. 
#print(z500_mslp_arr.shape)
z500_mslp_arr = temp_array.reshape(nday, -1)        #reshape 
print(z500_mslp_arr.shape)

#array_4d_restored = prova_arr.reshape(nday, nlat, nlon, 2) # to restore the dimensionality

max_value=-9999999
min_value=999999   #we are setting the min value and max value variables to a value so that there is no junk in the variable and each will easily overcome the set value.

for i in range(nday):
    min_value=min(min_value,np.min(z500_mslp_arr[i,:]))
    max_value=max(max_value,np.max(z500_mslp_arr[i,:]))
print("max: ", max_value, "; min: ", min_value)

#We are generating the MSLP factor to be multipled to the data to normalize it
norm_factor=100./(max_value-min_value)
print("norm_factor: " + str(norm_factor))

#The data is now being normalized.
arr_norm = z500_mslp_arr*norm_factor


data_train = arr_norm[:train_perc]
data_test = arr_norm[train_perc:]

print("Checking the train and test set ")
print("lenght of all norm. data:" + str(len(arr_norm)))
print("lenght of norm. train data:" + str(len(data_train)))
print("lenght of norm. test data:" + str(len(data_test)))

print( dataset_str + '_' + domain_region + '_Z500_MSLP_som_data_train_anomalies_' + EU_domain + '_45rm.npy')

mslp_raw.to_netcdf(output_path +  dataset_str + '_' + domain_region + '_' + som_variables_str + '_som_raw_anomalies_' + EU_domain + '_45rm.nc')
np.save(output_path + dataset_str + '_' + domain_region + '_' + som_variables_str + '_som_all_data_anomalies_' + EU_domain + '_45rm.npy', arr_norm)
np.save(output_path + dataset_str +'_' + domain_region + '_' + som_variables_str + '_som_data_train_anomalies_' + EU_domain + '_45rm.npy', data_train)
np.save(output_path + dataset_str +'_' + domain_region + '_' + som_variables_str + '_som_data_test_anomalies_' + EU_domain + '_45rm.npy', data_test)
np.save(output_path + dataset_str +'_' + domain_region + '_' + som_variables_str + '_som_time_data_anomalies_' + EU_domain + '_45rm.npy', time_values)

############################################################
#%% Preparing the SOM with precipitation
## SOM - only pr

def select_latlon(ds):
    #return ds.sel(lat = slice(60,36), lon = slice(-10,19)) #EU_1
    #return ds.sel(lat = slice(60,35), lon = slice(-15,30))#EU_2
    #return ds.sel(lat = slice(60,25), lon = slice(-13,35))#EU_3 (aggiornato)
    return ds.sel(lat = slice(36,48), lon = slice(5,19))# Italy
    #return ds.sel(lat = slice(47,25), lon = slice(-13,35))#EU_3_MEDITERRANEO -> EU4
#34N - 48M   4E - 16E
EU_domain = "EU3"

dataset_str = extreme_list_str

path_CERRA = "C:/Users/Criss/Documents/Lavoro/Assegno_2024_2025/Dati/CERRA_LAND"            # path 
#pr_file = path_MSWEP + "/MSWEP_1991_2020_Italy.nc"

pr_file = path_CERRA + "/All_precip_CERRA_LAND_1984_2021_italy.nc"

dy = xr.open_mfdataset(pr_file, preprocess=select_latlon)
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

mask_file = path_CERRA + "/prova_mask.nc"
dy = xr.open_mfdataset(mask_file, preprocess=select_latlon)
print(dy)

mask_lon = dy['lon']
mask_lat = dy['lat']
mask_nx = int((mask_lat.size))
mask_ny = int((mask_lon.size))

# ok sono stessa dimensione ora
mask = dy["stl1"].values
prova_mask = mask/mask

mask = dy["stl1"].values
prova_mask = mask/mask

pr_mask = pr_CERRA * prova_mask
pr_raw_CERRA = pr_raw_CERRA * prova_mask
#pr_CERRA = pr_CERRA * prova_mask

fig = plt.figure(figsize=(12, 8))    
fig.suptitle("Domain", fontsize=20)

# Figure - base
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
im_0 = plt.contourf(pr_lon, pr_lat, pr_mask[0])
ax.add_feature(cfeature.COASTLINE)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False, linewidth=0.33, color='k', alpha=0.5)
gl.right_labels = gl.top_labels = False


#%Train test split
print("lenght of all data:" + str(len(pr_time_CERRA)))
train_perc = round(len(pr_time_CERRA)*70/100)
print("lenght of train data:" + str(train_perc))
test_perc = len(pr_time_CERRA) - train_perc
print("lenght of test data:" + str(test_perc))

nday =int((ds['time'].size))
nlat = int((ds['lat'].size))
nlon = int((ds['lon'].size))


print(pr_raw_CERRA.shape)

#generate the empty array that will house the 6-hour interval data.

mslparr = np.empty((nday, nlat*nlon))  #This is the new array that we will place the data into. 

#We are now going to place the raw MSLP data into the array (mslparr)
#for i in range(nday):
#    mslparr[i,:]= pr_raw_CERRA[i,:,:].stack(point=["lat", "lon"])

mslparr = pr_CERRA.reshape(nday, -1)        #reshape 
print(mslparr.shape)

maxmslp=-9999999
minmslp=999999   #we are setting the minmslp and maxmslp variables to a value so that there is no junk in the variable and each will easily overcome the set value.

for i in range(nday):
    minmslp=min(minmslp,np.nanmin(mslparr[i,:]))
    maxmslp=max(maxmslp,np.nanmax(mslparr[i,:]))
print(maxmslp, minmslp)

#We are generating the MSLP factor to be multipled to the data to normalize it
mslp_factor=100./(maxmslp-minmslp)
print(mslp_factor)

#The data is now being normalized.
arr_norm = mslparr*mslp_factor
arr_norm.shape


data_train = arr_norm[:train_perc]
data_test = arr_norm[train_perc:]


print("Checking the train and test set ")
print("lenght of all norm. data:" + str(len(arr_norm)))
print("lenght of norm. train data:" + str(len(data_train)))
print("lenght of norm. test data:" + str(len(data_test)))

print( dataset_str + '_' + domain_region + '_pr_som_data_train_anomalies_' + EU_domain + '_45rm_nomask.npy')

pr_raw_CERRA.to_netcdf(output_path +  dataset_str + '_' + domain_region + '_pr_som_raw_anomalies_' + EU_domain + '_45rm_nomask.nc')
np.save(output_path + dataset_str + '_' + domain_region + '_pr_som_all_data_anomalies_' + EU_domain + '_45rm_nomask.npy', arr_norm)
np.save(output_path + dataset_str +'_' + domain_region + '_pr_som_data_train_anomalies_' + EU_domain + '_45rm_nomask.npy', data_train)
np.save(output_path + dataset_str +'_' + domain_region + '_pr_som_data_test_anomalies_' + EU_domain + '_45rm_nomask.npy', data_test)
np.save(output_path + dataset_str +'_' + domain_region + '_pr_som_time_data_anomalies_' + EU_domain + '_45rm_nomask.npy', pr_time_CERRA)


#%% SOM Z500 and pr
def select_latlon(ds):
    #return ds.sel(lat = slice(60,36), lon = slice(-10,19)) #EU_1
    #return ds.sel(lat = slice(60,35), lon = slice(-15,30))#EU_2
    return ds.sel(lat = slice(60,25), lon = slice(-13,35))#EU_3 (aggiornato)
    #return ds.sel(lat = slice(47,25), lon = slice(-13,35))#EU_3_MEDITERRANEO -> EU4
#34N - 48M   4E - 16E
EU_domain = "EU3"

dataset_str = extreme_list_str
som_variables_str = "Z500_pr"

name_file_z500 = "ERA5_daily_mean_Geop_500hPa_1985_2019_" + EU_domain + "_anomalies.nc" 
dy = xr.open_mfdataset(data_path_ERA5 + name_file_z500,    #Xarray features will be used throughout this tutorial 
                         preprocess=select_latlon)
ds = dy.sel(time=extreme_list, method="nearest") 
#print(ds)

time_values = ds['time'].values
z_values = ds['Z'].values
z_values = z_values / g
z_raw = (ds['Z'])/g  #This is the data the NON-anomaly data.
lon = ds['lon'].values
lat = ds['lat'].values
z_raw = z_raw[:,0,:,:]

fig = plt.figure(figsize=(12, 8))    
fig.suptitle("Domain", fontsize=20)

# Figure - base
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
im_0 = plt.contourf(lon, lat, z_values[0,0])
ax.add_feature(cfeature.COASTLINE)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False, linewidth=0.33, color='k', alpha=0.5)
gl.right_labels = gl.top_labels = False


#%Train test split
print("lenght of all data:" + str(len(time_values)))
train_perc = round(len(time_values)*70/100)
print("lenght of train data:" + str(train_perc))
test_perc = len(time_values) - train_perc
print("lenght of test data:" + str(test_perc))

nday =int((ds['time'].size))
nlat = int((ds['lat'].size))
nlon = int((ds['lon'].size))

#%%
def select_latlon(ds):
    #return ds.sel(lat = slice(60,36), lon = slice(-10,19)) #EU_1
    #return ds.sel(lat = slice(60,35), lon = slice(-15,30))#EU_2
    #return ds.sel(lat = slice(60,25), lon = slice(-13,35))#EU_3 (aggiornato)
    return ds.sel(lat = slice(36,48), lon = slice(5,19))# Italy
    #return ds.sel(lat = slice(47,25), lon = slice(-13,35))#EU_3_MEDITERRANEO -> EU4
    
# carico i dati
path_CERRA = "C:/Users/Criss/Documents/Lavoro/Assegno_2024_2025/Dati/CERRA_LAND"            # path 
#pr_file = path_MSWEP + "/MSWEP_1991_2020_Italy.nc"

pr_file = path_CERRA + "/All_precip_CERRA_LAND_1984_2021_italy.nc"

dy = xr.open_mfdataset(pr_file, preprocess=select_latlon)
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
fig = plt.figure(figsize=(12, 8))    
fig.suptitle("Domain", fontsize=20)

# Figure - base
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
im_0 = plt.contourf(pr_lon, pr_lat, pr_CERRA[0])
ax.add_feature(cfeature.COASTLINE)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False, linewidth=0.33, color='k', alpha=0.5)
gl.right_labels = gl.top_labels = False

## non usiamo la maschera
#%%
print("ERA5")
print(nlat, nlon)

print("pr")
print(pr_nx, pr_ny)

print(pr_raw_CERRA.shape)
# we want to expand pr_matrix in order to have the same dimension as ERA5
# therefore we create a matrix full of zeros
new_shape = (len(pr_raw_CERRA), nlat, nlon)
pr_new_matrix = np.zeros(new_shape)

# Let's copy the data 
pr_new_matrix[:pr_raw_CERRA.shape[0], :pr_raw_CERRA.shape[1], :pr_raw_CERRA.shape[2]] = pr_raw_CERRA

print(pr_new_matrix.shape)  

#%%

print(z_raw.shape, pr_new_matrix.shape)

temp_array = np.stack([z_raw, pr_new_matrix], axis=-1)   #stack into new axis
temp_array.shape
#z500_mslp_arr = np.empty((nday, nlat*nlon*2))      #This is the new array that we will place the data into. 
#print(z500_mslp_arr.shape)
z500_pr_arr = temp_array.reshape(nday, -1)        #reshape 
print(z500_pr_arr.shape)

max_value=-9999999
min_value=999999   #we are setting the min value and max value variables to a value so that there is no junk in the variable and each will easily overcome the set value.

for i in range(nday):
    min_value=min(min_value,np.min(z500_pr_arr[i,:]))
    max_value=max(max_value,np.max(z500_pr_arr[i,:]))
print("max: ", max_value, "; min: ", min_value)

#We are generating the MSLP factor to be multipled to the data to normalize it
norm_factor=100./(max_value-min_value)
print("norm_factor: " + str(norm_factor))

#The data is now being normalized.
arr_norm = z500_pr_arr*norm_factor


data_train = arr_norm[:train_perc]
data_test = arr_norm[train_perc:]

print("Checking the train and test set ")
print("lenght of all norm. data:" + str(len(arr_norm)))
print("lenght of norm. train data:" + str(len(data_train)))
print("lenght of norm. test data:" + str(len(data_test)))

print( dataset_str + '_' + domain_region + '_Z500_MSLP_som_data_train_anomalies_' + EU_domain + '_45rm.npy')

#%%
z_raw.to_netcdf(output_path +  dataset_str + '_' + domain_region + '_' + som_variables_str + '_Z500_raw_' + EU_domain + '_45rm.nc')
pr_raw_CERRA.to_netcdf(output_path +  dataset_str + '_' + domain_region + '_' + som_variables_str + '_pr_raw_' + EU_domain + '_45rm.nc')
np.save(output_path + dataset_str + '_' + domain_region + '_' + som_variables_str + '_som_all_data_anomalies_' + EU_domain + '_45rm.npy', arr_norm)
np.save(output_path + dataset_str +'_' + domain_region + '_' + som_variables_str + '_som_data_train_anomalies_' + EU_domain + '_45rm.npy', data_train)
np.save(output_path + dataset_str +'_' + domain_region + '_' + som_variables_str + '_som_data_test_anomalies_' + EU_domain + '_45rm.npy', data_test)
np.save(output_path + dataset_str +'_' + domain_region + '_' + som_variables_str + '_som_time_data_anomalies_' + EU_domain + '_45rm.npy', time_values)


#%%
mask_file = path_CERRA + "/prova_mask.nc"
dy = xr.open_mfdataset(mask_file, preprocess=select_latlon)
print(dy)

mask_lon = dy['lon']
mask_lat = dy['lat']
mask_nx = int((mask_lat.size))
mask_ny = int((mask_lon.size))
print(mask_nx, mask_ny)
#%%
# ok sono stessa dimensione ora
mask = dy["stl1"].values
prova_mask = mask/mask

mask = dy["stl1"].values
prova_mask = mask/mask

pr_mask = pr_CERRA * prova_mask
pr_raw_CERRA = pr_raw_CERRA * prova_mask
#pr_CERRA = pr_CERRA * prova_mask

fig = plt.figure(figsize=(12, 8))    
fig.suptitle("Domain", fontsize=20)

# Figure - base
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
im_0 = plt.contourf(pr_lon, pr_lat, pr_mask[0])
ax.add_feature(cfeature.COASTLINE)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False, linewidth=0.33, color='k', alpha=0.5)
gl.right_labels = gl.top_labels = False


pr_mask[np.isnan(pr_mask)] = 0


fig = plt.figure(figsize=(12, 8))    
fig.suptitle("Domain", fontsize=20)

# Figure - base
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
im_0 = plt.contourf(pr_lon, pr_lat, pr_mask[0])
ax.add_feature(cfeature.COASTLINE)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False, linewidth=0.33, color='k', alpha=0.5)
gl.right_labels = gl.top_labels = False

 
#%%


#%%
#array_4d_restored = prova_arr.reshape(nday, nlat, nlon, 2) # to restore the dimensionality


