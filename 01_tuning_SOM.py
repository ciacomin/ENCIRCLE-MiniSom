# -*- coding: utf-8 -*-
"""
@author: ciacomin

# This file manages the tuning step

"""
import os
# Fundamental libreries
import numpy as np
import pandas as pd
import glob
from enum import Enum
import xarray as xr
import winsound

os.chdir('C:\\your_directory') # if you need to change to your directory 

#from functions_new_version import *
from function_SOMs import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from itertools import product

#import for the SOM
import minisom
from minisom import asymptotic_decay
import datetime
from datetime import date
import pickle

from sklearn.metrics import silhouette_score

import pymannkendall as mk
from scipy.stats import linregress

class RegionDomain(Enum):
    NORTH_ITALY = "north"
    ITALY = "italy"
    
class Dataset(Enum):
    CERRA_LAND = "cerra-land6"
    CERRA = "cerra6"
    ARCIS = "arcis3"
    MSWEP = "mswep"
    
class DatasetString(Enum):
    CERRA_LAND = "CERRA_LAND"
    CERRA = "CERRA"
    ARCIS = "ARCIS"
    MSWEP = "MSWEP"
    
def select_latlon(ds):
    # this function selects the domain for the SOMs
    
    return ds.sel(lat = slice(52,27), lon = slice(-13,35)) # 

# PATHS
path_extremes="data/" # where you have saved the csv files of the extreme
output_path ="output_SOM/" # the output path
folderpath_tuning = 'tuningFinale/'  # Output of tuning section, if you want to save your results

# datasets
data_path_ERA5 = "data/ERA5/" 

################################################################
#%% 0) Dataset and variable selection 
# Chose first which dataset and domain region
dataset_extreme = Dataset.CERRA.value                      # chose the desired dataset
pr_dataset_str = DatasetString.CERRA.value                 # same as before 
region_domain = RegionDomain.ITALY.value                   # chose the desired region domain
# Chose the domain for the SOM
SOM_domain = "EU5"                                         # chose the desired SOM domain

start_month = 11
start_year = 1984
start_month_str = "Nov"

end_month = 10
end_year = 2024
end_month_str = "Oct"

start_date = datetime.date(start_year, start_month, 1)
end_date = datetime.date(end_year, end_month, 31) 

print("from ", start_date, " to ", end_date)

n_WA = 0
if region_domain == "north":
    n_WA = 94
elif region_domain == "italy":
    n_WA = 156
else :
    print("please check your region_domain")

#%% 1) Compute the extreme_pr_days_list
pr_days_dataset_filepath = glob.glob(path_extremes + dataset_extreme + "*" + region_domain + "*" + "withArea.xlsx" )[0]
print(pr_days_dataset_filepath)

print("1) DATASET AND DOMAIN REGION SELECTION ")
print("   I am computing the extremes for " + dataset_extreme + " from " + start_month_str + " " + str(start_year) + " to " + end_month_str + " " + str(end_year))
# here you can select the time period you are interested into
raw_pr_days_df, cut_pr_days_df = dataframe_cut_xlsx_month(pr_days_dataset_filepath, start_year, end_year, start_month, end_month, True)  
# here you can select a spatial filter 
extreme_pr_days_df = dataframe_extremes_xlsx(cut_pr_days_df, "intense rain day (1000km2)")           #here you can change the filter

# Getting the extreme days
extreme_pr_days = extreme_pr_days_df["Time"] # Series/Dataframe object
extreme_pr_days_list = extreme_pr_days.tolist()       # List object 
print("   number of extremes: ", str(len(extreme_pr_days_list)))

# if you want to save them 
print("I am saving "+ pr_dataset_str + '_extreme_dates_' + region_domain + "_" + str(start_month) + "-" + str(start_year) + "_" + str(end_month) + "-" + str(end_year) + '.npy')
np.save(output_path + pr_dataset_str + '_extreme_dates_' + region_domain + "_" + str(start_month) + "-" + str(start_year) + "_" + str(end_month) + "-" + str(end_year) + '.npy', np.array(extreme_pr_days_list, dtype=object), allow_pickle=True)

#%% 1.1) if you want directly to load the extremes uncomment
# If you have a list of extreme days you can select them directly 
# If you are on Spyder, comment with Ctrl + 4 and uncomment with Ctrl + 5

# =============================================================================
# print("I'm loading the extremes of " + pr_dataset_str + '_extreme_dates_' + region_domain + "_" + str(start_month) + "-" + str(start_year) + "_" + str(end_month) + "-" + str(end_year) + '.npy')
# extreme_pr_days_list = np.load(output_path + pr_dataset_str + '_extreme_dates_' + region_domain + "_" + str(start_month) + "-" + str(start_year) + "_" + str(end_month) + "-" + str(end_year) + '.npy', allow_pickle=True)
# print(extreme_pr_days_list.shape)
# =============================================================================

#%%
######################################################
################ LOADING THE DATA ####################
######################################################

# Here you 
# 2a) load Z500, detrend it and apply latitude weights
# 2b) load mSLP and apply latitude weights

print("   SOM domain: "+ SOM_domain)

# 2a) load the variable of interest
print(" -Loading the variables-")
print("   Loading Z500...")
name_file_z500 = "ERA5_daily_mean_Geop_500hPa_1984_2024_EU3_anomalies_clim1985_2019.nc"           # Anomalies of Z500
dy_z500 = xr.open_mfdataset(data_path_ERA5 + name_file_z500, preprocess=select_latlon)   # Here you select the domain

### Detrending Z500
time_values = dy_z500['time'].values
z_values = dy_z500['Z'].values[:,0,:,:] / g 
nday, nlat, nlon = z_values.shape
lon = dy_z500['lon'].values  #they are the same for the two variables 
lat = dy_z500['lat'].values

### Latitude Weights
# weigths associated with latitudes, np.cos wants radiants
# so first you need to convert them in radiants
lat_rad = np.radians(lat)
cos_lat = np.cos(lat_rad)
sum_cos_lat = np.sum(cos_lat)
weights_lat = np.sqrt(cos_lat / sum_cos_lat)  
#weights_lat = np.ones(101)
print(weights_lat.shape)

weights_lat_trend = cos_lat / sum_cos_lat

weights_matrix_trend = np.tile(weights_lat_trend[:, np.newaxis], (1, 193))
#diff = weights_matrix[:,10] - weights_lat #should be all 0s

z_values_trend = np.multiply(z_values, weights_matrix_trend)

# Reshape and detrend
z500_2d = z_values_trend.reshape(nday, -1)
z500_mean_values = np.mean(z500_2d, axis=1)

z500_slope, z500_intercept, _, _, _ = linregress(np.arange(nday), z500_mean_values)
z500_trend = z500_slope * np.arange(nday) + z500_intercept
z_values_detrended = z_values - z500_trend[:, np.newaxis, np.newaxis]

dy_z500['Z_detrended'] = (('time','lat','lon'), z_values_detrended)

# Here you select the extreme days
ds = dy_z500.sel(time=extreme_pr_days_list, method="nearest")          

time_values = ds['time'].values
z_values = ds['Z_detrended'].values

#%%
# =============================================================================
# 2b) loading mSLP
print("   Loading mslp...")
name_file_mslp = "ERA5_daily_mean_mSLP_sfc_1984_2024_EU3_anomalies_clim1985_2019.nc"
dy_mslp = xr.open_mfdataset(data_path_ERA5 + name_file_mslp,    #Xarray features will be used throughout this tutorial 
                         preprocess=select_latlon)
ds = dy_mslp.sel(time=extreme_pr_days_list, method="nearest") 
#ds = dy
#print(ds)

#time_values = ds['time'].values    #should be the same
mslp_values = ds['MSL'].values
mslp_values = mslp_values /100

#weights_matrix = np.tile(weights_lat, 193).reshape((101,193))
# do not use reshape since it change the values a little

weights_matrix = np.tile(weights_lat[:, np.newaxis], (1, 193))
#diff = weights_matrix[:,10] - weights_lat #should be all 0s

z_values = np.multiply(z_values, weights_matrix)
mslp_values = np.multiply(mslp_values, weights_matrix) #weights
print("     Z500: ", z_values.shape)
print("     mSLP: ", mslp_values.shape)

som_variables_str = "Z500_mSLP"

#%%
######################################################
############# PREPROCESSING THE DATA #################
######################################################

# Here you
# 3a) do train - test split,
# 3b) Standard Scaler fit on training set, transform and reshape
# 3c) stack variables and perform PCA

print(" -Preprocessing data train and test split-")
# 3a) data train - test SPLIT
print("   lenght of all data:" + str(len(time_values)))
train_perc = round(len(time_values)*70/100)
print("   lenght of train data:" + str(train_perc))
test_perc = len(time_values) - train_perc
print("   lenght of test data:" + str(test_perc))

# TRAIN DATASET 
z500_train = z_values[:train_perc]
mslp_train = mslp_values[:train_perc]
nday_train = len(z500_train)

# TEST DATASET
z500_test = z_values[train_perc:]
mslp_test = mslp_values[train_perc:]
nday_test = len(z500_test)

# EXTREME DATASET (all data)
z500_extremes = z_values
mslp_extremes = mslp_values
nday_extremes = len(z500_extremes)

# 3b) Standard Scaler fit on training set 
print("1", z500_train.shape)
scaler_z500 = StandardScaler().fit(z500_train.reshape(len(z500_train), -1))
scaler_mslp = StandardScaler().fit(mslp_train.reshape(len(mslp_train), -1))

z500_train_scaled = scaler_z500.transform(z500_train.reshape(len(z500_train), -1))
mslp_train_scaled = scaler_mslp.transform(mslp_train.reshape(len(mslp_train), -1))

z500_test_scaled = scaler_z500.transform(z500_test.reshape(len(z500_test), -1))
mslp_test_scaled = scaler_mslp.transform(mslp_test.reshape(len(mslp_test), -1))

z500_extremes_scaled = scaler_z500.transform(z500_extremes.reshape(len(z500_extremes), -1)) # all data - extremes 
mslp_extremes_scaled = scaler_mslp.transform(mslp_extremes.reshape(len(mslp_extremes), -1))

# reshape 
z500_train_scaled = z500_train_scaled.reshape(nday_train, nlat, nlon)
mslp_train_scaled = mslp_train_scaled.reshape(nday_train, nlat, nlon)

z500_test_scaled = z500_test_scaled.reshape(nday_test, nlat, nlon)
mslp_test_scaled = mslp_test_scaled.reshape(nday_test, nlat, nlon)

z500_extremes_scaled = z500_extremes_scaled.reshape(nday_extremes, nlat, nlon)
mslp_extremes_scaled = mslp_extremes_scaled.reshape(nday_extremes, nlat, nlon)

# 3c) stack variables and perform PCA 
print("-Stacking and performing PCA")
train_combined = np.stack([z500_train_scaled, mslp_train_scaled], axis=-1)
test_combined = np.stack([z500_test_scaled, mslp_test_scaled], axis=-1)
extremes_combined = np.stack((z500_extremes_scaled, mslp_extremes_scaled), axis=-1)

train_combined = train_combined.reshape(nday_train, -1)
test_combined = test_combined.reshape(nday_test, -1)
extremes_combined = extremes_combined.reshape(nday_extremes, -1)

print(extremes_combined.shape)

# PCA 
print(" -PCA-")
# Cumulative Explained Variance vs Number of Components
pca = PCA()
pca.fit(train_combined)
# Calculate cumulative explained variance
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

# Plot cumulative explained variance
plt.figure(figsize=(8,6), dpi=200)
plt.plot(np.arange(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance vs Number of Components')
plt.grid(True)
plt.xlim(0,50)
#plt.savefig("tuningFinale/0_CumulativeExplainedVariance"+ pr_dataset_str + ".png")
#plt.savefig("tuningFinale/0_CumulativeExplainedVariance" + pr_dataset_str + ".svg")
plt.show()

######################################################
#################### TUNING ##########################
######################################################
# first is to assess the best function, then sigma and LR

# Explained variance vs QE vs Neighborhood function
som_col = 3
som_row = 2
som_shape =(som_row, som_col)
min_som = min(som_col, som_row)
sigma = min_som -1        #The sigma value must be y-1. ù
learning_rate = 0.0005  #Learning Rate 
q_win = 100000.
q_error_list_cicle = []
t_error_list_cicle = []
n_index_list = [0.95, 0.96, 0.97, 0.98, 0.99] # explained variance
neighborhood_function_list = ["bubble", "gaussian", "mexican_hat", "triangle"]

plt.figure(figsize=(10,6), dpi=200)
plt.title("PCA explained variance vs QE, LR: " + str(learning_rate) + " sigma: " + str(sigma))
plt.xlabel("PCA explained variance")
plt.ylabel("Quantization error")
    
for neighborhood_function_index in range(len(neighborhood_function_list)):
    neighborhood_function = neighborhood_function_list[neighborhood_function_index]
    print("neighborhood_function:", neighborhood_function)
    q_error_list = []
    t_error_list = []
    #for n_index in range(1,len(data_train_std)):
        
    for n_index in n_index_list:
        print("n_comp.: ", n_index)
        pca = PCA(n_components=n_index).fit(train_combined)
        data_train_pca = pca.transform(train_combined)
        data_test_pca = pca.transform(test_combined)
        extremes_pca = pca.transform(extremes_combined)
       
        input_length = len(data_train_pca[0])  #This is value is the the length of the latitude X longitude. It is the second value in the data_train.shape step. 
        
        # SOM
        era5_som = minisom.MiniSom(som_row, som_col, input_len = input_length, sigma = sigma, learning_rate=learning_rate, neighborhood_function=neighborhood_function, decay_function = asymptotic_decay)
        era5_som.random_weights_init(data_train_pca)
        # train som
        era5_som.train(data_train_pca, num_iteration=100000,random_order=True, verbose=True) 
        
        q_error = round(era5_som.quantization_error(data_test_pca),3) # we assess the q_error on the test set
        q_error_list += [q_error]
        t_error = round(era5_som.topographic_error(data_test_pca),3)
        t_error_list += [t_error]
        #if q_error < q_win:
        #    q_win = q_error
        
    legend_plot = str(neighborhood_function)
    plt.plot(n_index_list, q_error_list, label=legend_plot)
    plt.plot(n_index_list, q_error_list, "ko")
    plt.legend()
    
    q_error_list_cicle += [q_error_list]
    t_error_list_cicle += [t_error_list]
    
plt.savefig(folderpath_tuning + "1_" + pr_dataset_str + "_" + str(start_year) + "_" + str(end_year) + "_" + str(som_row) + "x" + str(col) + "_QE_test_variable_nf_LR0_0005_sigma1.svg", format="svg")


#%% Assesing sigma and LR

som_col = 2
som_row = 2
som_shape =(som_row, som_col)
min_som = min(som_col, som_row)
sigma_list = [0.9, 0.95, 1, 1.05, 1.1,] #2x2
# sigma_list = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2] #2X3
learning_rate_list=[0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002, 0.005, 0.01]

q_win = 100000.
q_error_list_cicle = []
t_error_list_cicle = []


plt.figure(figsize=(10,6))
plt.title("Sigma and LR vs QE")
plt.xlabel("Sigma")
plt.ylabel("Quantization error")
    

pca = PCA(n_components=0.95).fit(train_combined) # Keeps 95% of the variance
data_train_pca = pca.transform(train_combined)
data_test_pca = pca.transform(test_combined)
        

for l_rate_index in range(len(learning_rate_list)):
    
    learning_rate = learning_rate_list[l_rate_index]
    print("Learning rate:", learning_rate)
        
    q_error_list = []
    t_error_list = []
    #for n_index in range(1,len(data_train_std)):
        
    for sigma_rate_index in range(len(sigma_list)):
        sigma = sigma_list[sigma_rate_index]
        print("sigma:", sigma)
        
        input_length = len(data_train_pca[0])  #This is value is the the length of the latitude X longitude. It is the second value in the data_train.shape step. 
        
        # SOM
        era5_som = minisom.MiniSom(som_row, som_col, input_len = input_length, sigma = sigma, learning_rate=learning_rate, neighborhood_function='gaussian', decay_function = asymptotic_decay)
        era5_som.random_weights_init(data_train_pca)
        # train som
        era5_som.train(data_train_pca, num_iteration=100000,random_order=True, verbose=True)
        
        q_error = round(era5_som.quantization_error(data_test_pca),3)
        q_error_list += [q_error]
        t_error = round(era5_som.topographic_error(data_test_pca),3)
        t_error_list += [t_error]
        #if q_error < q_win:
        #    q_win = q_error
  
    legend_plot = "LR:" + str(learning_rate)
    plt.plot(sigma_list, q_error_list, label=legend_plot)
    plt.plot(sigma_list, q_error_list, "ko")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 13})
    plt.tight_layout()
    
    q_error_list_cicle += [q_error_list]
    t_error_list_cicle += [t_error_list]
    
plt.savefig(folderpath_tuning+ "2_" + pr_dataset_str + "_" + str(start_year) + "_" + str(end_year) + "_" + str(som_row) + "x" + str(col) + "_QE_test_LR_sigma_variable_gaussian.svg", format="svg")
plt.savefig(folderpath_tuning+ "2_" + pr_dataset_str + "_" + str(start_year) + "_" + str(end_year) + "_" + str(som_row) + "x" + str(col) + "_QE_test_LR_sigma_variable_gaussian.png")


## MY FINAL CONFIGURATION
# gaussian
# ARCIS, 2X2: sigma = 1, LR = 0.002
# CERRA, 3x2: sigma = 1, LR = 0.005



