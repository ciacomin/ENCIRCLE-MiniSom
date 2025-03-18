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

#%% Create the SOM
# we train on data_train_pca
# but we test the q_error on the data_test_pca

input_length = len(data_train_pca[0])
for i in range(number_of_soms):   #The number of SOMs that will be generated. 
    # initialize random weights
    era5_hourly_som1 = minisom.MiniSom(som_row, som_col, input_len = input_length, sigma = sigma, learning_rate=learning_rate, neighborhood_function='gaussian', decay_function = asymptotic_decay)
    era5_hourly_som1.random_weights_init(data_train_pca)
    # train som
    era5_hourly_som1.train(data_train_pca, num_iteration=100000,random_order=True, verbose=True)
    q_error = era5_hourly_som1.quantization_error(data_test_pca)
    
    #Add the details of the SOM settings into the name of the file so that you know what the SOM is showing.
    with open(folderpath_SOMs + som_names +str(i+1)+'.p', 'wb') as outfile: #this is how you save the file, the str(i) is a unique name
        pickle.dump(era5_hourly_som1, outfile)
    weights = era5_hourly_som1._weights
    q_error_list += [q_error]
    i+=1
    if q_error < q_win:
        q_win = q_error
        win_weights = era5_hourly_som1
        
print('\007')
#%%
names = ([os.path.splitext(os.path.split(x)[-1])[0] for x in glob.glob(folderpath_SOMs + som_names + '*')])
filepaths = glob.glob(folderpath_SOMs + som_names + '*')  #this is showing the path and the given file
print(names) # this can give you the order, the second one is number 10 not 2  

#%% Master SOM
# we want to operate just on the "best" or "Master" SOM. 
# So we need to calculate the performance on the test set
# we want to calculate QE and TE on both train and data test 

q_error_list_train = []
t_error_list_train = []

q_error_list_test = []
t_error_list_test = []

data_train_chosen = data_train_pca
data_test_chosen = data_test_pca
all_data_chosen = all_data_pca
for path, name in zip(filepaths, names):
    with open (path, 'rb') as f:
        file = pickle.load(f) #This is loading every single som in that location
        q_error_train = round(file.quantization_error(data_train_chosen),3) #this is grabbing every q error out to 3 decimal places
        t_error_train = round(file.topographic_error(data_train_chosen),3) #this is grabbing ever topographic error out to 3 decimal places
        q_error_list_train += [q_error_train]
        t_error_list_train += [t_error_train]
        
        q_error_test = round(file.quantization_error(data_test_chosen),3) #this is grabbing every q error out to 3 decimal places
        t_error_test = round(file.topographic_error(data_test_chosen),3) #this is grabbing ever topographic error out to 3 decimal places
        q_error_list_test += [q_error_test]
        t_error_list_test += [t_error_test]
        
mean_qerror_train = np.mean(q_error_list_train)
mean_qerror_test  = np.mean(q_error_list_test)

mean_terror_train = np.mean(t_error_list_train)
mean_terror_test  = np.mean(t_error_list_test)

index_best_QE = np.where(q_error_list_test == np.min(q_error_list_test))[0][0]
print("qerr. best is " + names[index_best_QE])
print(q_error_list_test)
print(" ")

index_best_TE = np.where(t_error_list_test == np.min(t_error_list_test))[0][0]
print("topoerr. best is " + names[index_best_TE])
print(t_error_list_test)
print(" ")

#%%
# t_error when using PCA is usually really small (but you can check)
# so we consider as best the SOM that has the minimum QE
name_best=names[index_best_QE]
master_som_name = name_best
print("The best is " + str(name_best))
som = pickle.load(open(filepaths[index_best_QE], 'rb'))
weights = som._weights
print(weights.shape)

# Reverse PCA to return to original features
data_prototypes_pca_inverse = pca.inverse_transform(weights)

# new dictionary for the new data
keys = [i for i in product(range(som_row), range(som_col))]  ## DIM OF SOMS
winmap = {key: [] for key in keys}

#%%
# We train the SOM on the train set, but we apply it to all 
data_chosen = all_data_pca
test_train_all = "all"

winner_coordinates = np.array([som.winner(x) for x in data_chosen]).T
        
# with np.ravel_multi_index we convert the bidimensional
# coordinates to a monodimensional index
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)
silhouette_score(data_chosen, labels=cluster_index, metric='euclidean') # if you want to check the silhouette_score

# =============================================================================
# plt.figure(figsize=(12,8))
# for c in np.unique(cluster_index):
#     plt.scatter(data_chosen[cluster_index == c, 0],
#         data_chosen[cluster_index == c, 1], label='cluster='+str(c+1), alpha=.7)
# 
# # plotting centroids
# for centroid in som.get_weights():
#     plt.scatter(centroid[:, 0], centroid[:, 1], marker='x', 
#                 s=20, linewidths=15, color='k', label='centroid')
# plt.title('Centroids (' + test_train_all + " set) " + som_variables_str + ". " + pr_dataset_str + " " + region_domain + " " + str(som_col) + "by" + str(som_row) + " n." + name_best[-2:])
# #plt.savefig(folderpath + pr_dataset_str + "_" + test_train_all + '_centroids_' +name_best+'.svg')
# #plt.savefig(folderpath + pr_dataset_str + "_" + test_train_all + '_centroids_' +name_best+'.png')
# plt.legend();
# =============================================================================

nx = int(len(lat))
ny = int(len(lon))

# just for graphic reason
wt_list = [[2, 1], [4, 3]] # ARCIS Central-North
wt_list = [[3, 4], [1, 2]] # CERRA Central-North 
wt_list = [[2, 4], [1, 3]] # MSWEP Central-North
wt_list = [[1, 3, 5], [2, 4, 6]] # CERRA Italy

# Correlation 
for i, x in enumerate(data_chosen):
    winmap[som.winner(x)].append(i)
    
corr_list = []
for k in range(weights.shape[0]):
    for j in range(weights.shape[1]):
        index_maps = winmap[(k,j)]
        #print(index_maps) #index of the maps of a single node
        cluster_maps = [data_chosen[i] for i in index_maps] # maps relative to such indices
        
        print("Node " + str(k*weights.shape[1] + j+1))
        print(" number of maps: " + str((len(cluster_maps))))
        corr_list_temp = []
        for i in range(len(cluster_maps)):
            corr_list_temp += [np.corrcoef(weights[k,j,:], cluster_maps[i])[0,1]]
    
        print(" corr.: " + str(np.mean(corr_list_temp))) 
        corr_list += [np.mean(corr_list_temp)]

        frequencies = som.activation_response(all_data_chosen)
        freq_perc = frequencies / len(all_data_chosen) * 100   # percentual freq


