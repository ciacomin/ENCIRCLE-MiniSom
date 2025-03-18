# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 15:28:54 2025

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
#from xarray import DataArray

os.chdir('C:\\Users\\Criss\\Documents\\Lavoro\\Assegno_2024_2025\\Codici') #if you need to change to your directory 

from functions_new_version import *
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

# datasets
data_path_ERA5 = "data/ERA5/" 

################################################################
#os.chdir('C:\\your_directory') # if you need to change to your directory 
#%% 0) Dataset and variable selection 
# Chose first which dataset and domain region
dataset = Dataset.CERRA_LAND.value                      # chose the desired dataset
pr_dataset_str = DatasetString.CERRA_LAND.value         # same as before 
region_domain = RegionDomain.ITALY.value                # chose the desired region domain
# Chose the domain for the SOM
SOM_domain = "EU5" #SOM domain

start_year = 1985
end_year = 2019 

#%% 1) Compute the extreme_pr_days_list
pr_days_dataset_filepath = glob.glob(path_extremes + dataset + "*" + region_domain + "*" + "withArea.xlsx" )[0]

print("1) DATASET AND DOMAIN REGION SELECTION ")
print("   I am computing the extremes for " + dataset)
# here you can select the time period you are interested into
raw_pr_days_df, cut_pr_days_df = dataframe_cut_xlsx(pr_days_dataset_filepath, start_year, end_year, True)      # we want 1985 - 2019 period
# here you can select a spatial filter 
extreme_pr_days_df = dataframe_extremes_xlsx(cut_pr_days_df, "intense rain day (1000km2)")           #here you can change the filter

# Getting the extreme days
extreme_pr_days = extreme_pr_days_df["Time"] # Series/Dataframe object
extreme_pr_days_list = extreme_pr_days.tolist()       # List object 
print("   number of extremes: ", str(len(extreme_pr_days_list)))

# if you want to save them 
#print("I am saving "+ extreme_list_str + '_extreme_dates_' + domain_region + '.npy')
#np.save(output_path + extreme_list_str + '_extreme_dates_' + domain_region + '.npy', np.array(extreme_pr_days_list, dtype=object), allow_pickle=True)

#%% 1.1) if you want directly to load the extremes uncomment
# If you have a list of extreme days you can select them directly 
# If you are on Spyder, comment with Ctrl + 4 and uncomment with Ctrl + 5
# =============================================================================
# extreme_list_str = "ARCIS"  
# #extreme_list_str = "MSWEP"
# #extreme_list_str = "CERRA_LAND"
# 
# # domain
# #domain_region = "Italy"
# domain_region  = "North-Italy"
# 
# print("I'm loading the extremes of "+ extreme_list_str + '_extreme_dates_' + domain_region + '.npy')
# extreme_pr_days_list = np.load(output_path + extreme_list_str + '_extreme_dates_' + domain_region + '.npy', allow_pickle=True)
# print(extreme_pr_days_list.shape)
# =============================================================================

#%% 2) Preparing the SOM - Z500 and mSLP or Z500 and TCWV

# Here you choose the domain for the SOM and preparing the files you need
# 2a) load the variable of interest and do the standardization (per each variable)
# 2b) start the pre-processing with the data train - test split
# 2c) PCA
# 2d) save your files, if needed (uncomment)

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

nday =int((ds['time'].size))  # number of extreme days
nlat = int((ds['lat'].size))
nlon = int((ds['lon'].size))

#  reshape and standardization ( new_x = x - x_mean / standard_deviation )
z500_2d = z_values.reshape(nday, -1)        #reshape 
scaler_z500 = StandardScaler()
z_values_scaled = scaler_z500.fit_transform(z500_2d)

#%%
# =============================================================================

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

#%%
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

#%%
## 2c) PCA
print(" -PCA-")
# Cumulative Explained Variance vs Number of Components
# Uncomment 
pca = PCA()
pca.fit(data_train)
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

#%%
# pca on train set
folderpath_tuning = 'tuningFinale/'  # Output of tuning section, if you want to save your resulrs

# first is to assess the best function, then sigma and LR

# Explained variance vs QE vs Neighborhood function
som_col = 2
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
        pca = PCA(n_components=n_index) # Keeps 99% of the variance
        data_train_pca = pca.fit_transform(data_train)
        data_test_pca = pca.transform(data_test)
        #all_data_pca = pca.transform(all_data_std)
        
        data_chosen = data_train_pca
        
        input_length = len(data_chosen[0])  #This is value is the the length of the latitude X longitude. It is the second value in the data_train.shape step. 
        
        # SOM
        era5_hourly_som1 = minisom.MiniSom(som_row, som_col, input_len = input_length, sigma = sigma, learning_rate=learning_rate, neighborhood_function=neighborhood_function, decay_function = asymptotic_decay)
        era5_hourly_som1.random_weights_init(data_chosen)
        # train som
        era5_hourly_som1.train(data_chosen, num_iteration=100000,random_order=True, verbose=True) 
        
        q_error = round(era5_hourly_som1.quantization_error(data_test_pca),3) # we assess the q_error on the test set
        q_error_list += [q_error]
        t_error = round(era5_hourly_som1.topographic_error(data_test_pca),3)
        t_error_list += [t_error]
        #if q_error < q_win:
        #    q_win = q_error
        
    index_first_min =0
    for i in range(1, len(q_error_list)):
        if q_error_list[i] < q_error_list[i-1]:
            index_first_min = i
            break
        
    legend_plot = str(neighborhood_function)
    plt.plot(n_index_list, q_error_list, label=legend_plot)
    plt.plot(n_index_list, q_error_list, "ko")
    plt.legend()
    
    q_error_list_cicle += [q_error_list]
    t_error_list_cicle += [t_error_list]
    
plt.savefig(folderpath_tuning + "1_" + pr_dataset_str + "_QE_test_variable_nf_LR0_0005_sigma1.svg", format="svg")

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
    
pca = PCA(n_components=0.95) # Keeps 95% of the variance
data_train_pca = pca.fit_transform(data_train)
data_test_pca = pca.transform(data_test)
        

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
        era5_hourly_som1 = minisom.MiniSom(som_row, som_col, input_len = input_length, sigma = sigma, learning_rate=learning_rate, neighborhood_function='gaussian', decay_function = asymptotic_decay)
        era5_hourly_som1.random_weights_init(data_train_pca)
        # train som
        era5_hourly_som1.train(data_train_pca, num_iteration=100000,random_order=True, verbose=True)
        
        q_error = round(era5_hourly_som1.quantization_error(data_test_pca),3)
        q_error_list += [q_error]
        t_error = round(era5_hourly_som1.topographic_error(data_test_pca),3)
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
    
plt.savefig(folderpath+ "2_" + pr_dataset_str + "_" + str(som_row) + "x" + str(col) + "_QE_test_LR_sigma_variable_gaussian.svg", format="svg")
plt.savefig(folderpath+ "2_" + pr_dataset_str + "_" + str(som_row) + "x" + str(col) + "_QE_test_LR_sigma_variable_gaussian.png")




