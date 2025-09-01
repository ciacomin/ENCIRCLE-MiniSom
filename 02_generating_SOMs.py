"""
@author: ciacomin

# This file manages the generation of the SOMs
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
    
    return ds.sel(lat = slice(52,27), lon = slice(-13,35)) 
    
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
# ARCIS, 2x2: sigma = 1, LR = 0.002
# CERRA, 3x2: sigma = 1, LR = 0.005

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
perc_pca = 0.95
pca = PCA(n_components=perc_pca).fit(train_combined)
train_pca = pca.transform(train_combined)
test_pca = pca.transform(test_combined)
extremes_pca = pca.transform(extremes_combined)

#%% INITIALIZE THE SOM

######################################################
############## GENERATING THE SOMS ###################
######################################################

# Now that you have the setup you can create your SOMs
# ARCIS, 2x2: sigma = 1, LR = 0.002
# CERRA Land, 3x2: sigma = 1, LR = 0.005

#######
sigma = 0.95
learning_rate = 0.0008
#######

som_col = 3
som_row = 2
som_shape =(som_row, som_col)
min_som = min(som_col, som_row)
dim_som = som_col * som_row
number_of_soms = 10
q_win = 100000.
q_error_list = []
t_error_list = []

som_names =  pr_dataset_str + "_" + region_domain + "_" + SOM_domain + "_detrended_" + som_variables_str + "_PCA_" + str(perc_pca) + "_" + str(som_col) + "by" + str(som_row) + "_LR" + str(learning_rate) + "_sig" + str(sigma) + "_n_"
print(som_names)

#%% Create the SOM
# we train on train_pca
# but we test the q_error on the test_pca

input_length = len(train_pca[0])

for i in range(number_of_soms):   #The number of SOMs that will be generated. 
    # initialize random weights
    era5_som = minisom.MiniSom(som_row, som_col, input_len = input_length, sigma = sigma, learning_rate=learning_rate, neighborhood_function='gaussian', decay_function = asymptotic_decay)
    era5_som.random_weights_init(train_pca)
    # train som
    era5_som.train(train_pca, num_iteration=100000,random_order=True, verbose=True)
    q_error = era5_som.quantization_error(test_pca)
    
    #Add the details of the SOM settings into the name of the file so that you know what the SOM is showing.
    with open(folderpath_SOMs + som_names +str(i+1)+'.p', 'wb') as outfile: #this is how you save the file, the str(i) is a unique name
        pickle.dump(era5_som, outfile)
    weights = era5_som._weights
    q_error_list += [q_error]
    i+=1
    if q_error < q_win:
        q_win = q_error
        win_weights = era5_som
                
print('\007')

#%%
# %%
names = ([os.path.splitext(os.path.split(x)[-1])[0] for x in glob.glob(folderpath_SOMs + som_names + '*.p')])
filepaths = glob.glob(folderpath_SOMs + som_names + '*')  #this is showing the path and the given file
print(names) # this can give you the order, the second one is number 10 not 2  

### Master SOM
# we want to operate just on the "best" or "Master" SOM. 
# So we need to calculate the performance on the test set
# we want to calculate QE and TE on both train and data test 

q_error_list_train = []
t_error_list_train = []

q_error_list_test = []
t_error_list_test = []

for path, name in zip(filepaths, names):
    with open (path, 'rb') as f:
        file = pickle.load(f) #This is loading every single som in that location
        q_error_train = round(file.quantization_error(train_pca),3) #this is grabbing every q error out to 3 decimal places
        t_error_train = round(file.topographic_error(train_pca),3) #this is grabbing ever topographic error out to 3 decimal places
        q_error_list_train += [q_error_train]
        t_error_list_train += [t_error_train]
        
        q_error_test = round(file.quantization_error(test_pca),3) #this is grabbing every q error out to 3 decimal places
        t_error_test = round(file.topographic_error(test_pca),3) #this is grabbing ever topographic error out to 3 decimal places
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

name_best=names[index_best_QE]
master_som_name = name_best
print("The best is " + str(name_best))
master_som = pickle.load(open(filepaths[index_best_QE], 'rb'))
weights = master_som._weights
print(weights.shape)

# %%
######################################################
################# ANOMALIES ##########################
######################################################

data_prototypes_pca_inverse = pca.inverse_transform(weights)

# new dictionary for the new data
keys = [i for i in product(range(som_row), range(som_col))]  ## DIM OF SOMS
winmap = {key: [] for key in keys}

# We train the SOM on the train set, but we apply it to all data
data_chosen = extremes_pca
test_train_all = "all"

winner_extremes = np.array([master_som.winner(x) for x in data_chosen]).T
distances_extremes = master_som._distance_from_weights(data_chosen)
        
# with np.ravel_multi_index we convert the bidimensional
# coordinates to a monodimensional index
cluster_index_list = []
cluster_dates_list = []

cluster_index = np.ravel_multi_index(winner_extremes, som_shape)
cluster_index_list += [cluster_index] 
dim_som = som_col * som_row

# to get the dates 
cluster_nodes_dates = divide_dates_by_index(time_values, cluster_index, dim_som)

nx = int(len(lat))
ny = int(len(lon))

# just for graphic reason
wt_list = [[2, 4, 6], [1, 3, 5]] # CERRA Italy 1984 - 2024
#wt_list = [[1, 2], [3, 4]] # CERRA, ArCIS Central-North 
#wt_list = [[3, 1], [4, 2]] # MSWEP Central-North



i=1 
flat_wt_list = [item for sublist in wt_list for item in sublist]
#wt_list = [[3, 1], [4, 2]] # CERRA Central-North 

# Correlation 
for i, x in enumerate(data_chosen):
    winmap[master_som.winner(x)].append(i)
    
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

        frequencies = master_som.activation_response(data_chosen)
        freq_perc = frequencies / len(data_chosen) * 100   # percentual freq


# node 
fig, axs = plt.subplots(nrows=som_row,ncols=som_col, 
                        subplot_kw={'projection': ccrs.PlateCarree()},
                        figsize=(45, 15),facecolor='white') 
fig.tight_layout()
node = 0
for i in range(som_row):
    for j in range(som_col):
        
        frequencies = master_som.activation_response(data_chosen)
        freq_perc = frequencies / len(data_chosen) * 100   # percentual freq
        
        data_prototypes_node = data_prototypes_pca_inverse[i][j].reshape(1,-1) # mappa singolo nodo
                        
        # divido in Z500 e mslp
        SOM_array = data_prototypes_node.reshape(nx,ny,2)
        SOM_z500_temp = SOM_array[:,:,0]
        SOM_mslp_temp = SOM_array[:,:,1]
        
        SOM_z500_inverted = scaler_z500.inverse_transform(SOM_z500_temp.reshape(1,-1))
        SOM_mslp_inverted = scaler_mslp.inverse_transform(SOM_mslp_temp.reshape(1,-1))
        # ma devo fare l'inversione dello scaler
        
        SOM_z500 = SOM_z500_inverted.reshape(nx,ny)
        SOM_mslp = SOM_mslp_inverted.reshape(nx,ny) 
        
        # I need to remove the latitude weights
        print(SOM_z500.shape)
        SOM_z500 = np.divide(SOM_z500, weights_matrix)
        SOM_mslp = np.divide(SOM_mslp, weights_matrix)

        datacrs = ccrs.PlateCarree()        
        #axs=axs.flatten()
        levs = np.arange(-210, 211, 30)
        cs2=axs[i][j].contourf(lon, lat, SOM_z500,
                          transform = ccrs.PlateCarree(),
                          cmap="RdBu_r", levels = levs,extend='both')
        label_shaded = r"anom. $Z_{500}$ (m)"
        
        levels = np.arange(-20, 22, 2) # label height #mslp
        #levels = np.arange(-100, 100, 10)
        contour = axs[i][j].contour(lon, lat, SOM_mslp, levels, colors="red", transform = ccrs.PlateCarree(), linewidths=1)
        plt.clabel(contour, inline=True, fontsize=14, fmt='%1.0f') 
        
        axs[i][j].set_extent([lon[0], lon[-1], lat[0], lat[-1]], ccrs.PlateCarree())
        
        axs[i][j].coastlines()
        axs[i][j].add_feature(cfeature.BORDERS) 
        axs[i][j].set_title("WT" + str(wt_list[i][j]) + ": F.=%.2f" % freq_perc[i,j] + "%, " + r"$\rho$=%.2f" % corr_list[node] , fontsize=28)
        #axs[i][j].set_title("N" + str(node) + ": F.=%.2f" % freq_perc[i,j] + "%, " + r"$\rho$=%.2f" % corr_list[node] , fontsize=28)
        
        #axs[(k*4)+i].scatter(-156.36,71.19, c='yellow',marker= 'o',s=120, linewidth=2,edgecolors= "black" ,zorder= 4,transform=datacrs)
        
        
        # Title each subplot 
        #axs[0].set_title("F:" + str(int(frequencies[k,i])) + " (" + "%.2f" % freq_perc[k,i] + " %) " + r"$\rho$: %.2f" % corr_list[node] , fontsize=18)
        node = node + 1 

plt.tight_layout()
fig.subplots_adjust(bottom=0.25, top=0.9, left=0.05, right=0.6,
            wspace=0.05, hspace=0.25)

cbar_ax = fig.add_axes([0.08, 0.2, 0.5, 0.02])
#cbar=fig.colorbar(cs2,cax=cbar_ax, ticks = np.arange(lev_start, np.abs(lev_start)+lev_step, lev_step*2),orientation='horizontal')
cbar=fig.colorbar(cs2,cax=cbar_ax, ticks = levs,orientation='horizontal')

cbar.set_label(label_shaded, fontsize=26)
cbar.ax.tick_params(labelsize=22)
#cbar.set_label(r"anom. TCWV ($kg / m^2$)", fontsize=22)

#plt.savefig(folderpath_SOMs + pr_dataset_str + '_all_anomalyplot_' +name_best+'.svg', bbox_inches='tight')
#plt.savefig(folderpath_SOMs + pr_dataset_str + '_all_anomalyplot_' +name_best+'.png', bbox_inches='tight')
plt.show()

#%% Saving the dates in a csv files
######################################################
############## SAVING THE DATES ######################
######################################################

pr_days_dataset_filepath = glob.glob(path_extremes + dataset_extreme + "*" + region_domain + "*" + "withArea.xlsx" )[0]
raw_pr_days_df, cut_pr_days_df = dataframe_cut_xlsx(pr_days_dataset_filepath, start_year, end_year, True) 
cut_pr_days_pr_max_df = add_pr_max_column(cut_pr_days_df, 1, n_WA)  

df_file_path = folderpath_SOMs + "/" + pr_dataset_str + "_" + region_domain + "_" + str(start_year) + "_" + str(end_year) + "_" + str(som_col) + "by" + str(som_row) + "_sorted_by_dates_WT"
output_excel_file = df_file_path + ".xlsx" 

df_list = []

for i in range(len(cluster_nodes_dates)):
    print(len(cluster_nodes_dates[i]))

    # Convertiamo le date nel formato corretto
    cluster_dates_list_node = pd.to_datetime(cluster_nodes_dates[i]).normalize()  # normalize serve per perdere l'informazione sull'orario
    
    # Selezioniamo solo le righe con le date corrispondenti
    df_pr_selected = cut_pr_days_pr_max_df[cut_pr_days_pr_max_df["Time"].dt.normalize().isin(cluster_dates_list_node)].copy()  # .dt.normalize serve per perdere l'informazione sull'orario
    
    # Aggiungiamo le colonne aggiuntive
    #df_pr_selected["node"] = i + 1
    df_pr_selected["WT"] = flat_wt_list[i]  
    
    # Aggiungiamo alla lista
    df_list.append(df_pr_selected)

# Concatenare tutti i DataFrame in un unico DataFrame
df_final = pd.concat(df_list)

# Ordinare per la colonna "Time"
df_final = df_final.sort_values(by="Time", ascending=True)

# Scrivere su Excel in un unico foglio
with pd.ExcelWriter(output_excel_file) as writer:
    df_final.to_excel(writer, sheet_name="All_WTs", index=False)
    
df_final.to_csv(df_file_path + ".csv", index=False)
