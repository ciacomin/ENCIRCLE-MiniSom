"""
@author: ciacomin

Loading SOMs and performing analysis
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
path_ARCIS = "data/ArCIS"            
path_MSWEP = "data/MSWEP" 
path_CERRA = "path/CERRA_Land"   
path_CERRA_wa = "path/CERRA_Land_warning_region_mean" 

# shape_files
shapef_path= "data/shape_files/"  + 'ZA_2017_ID_v4_geowgs84.shp'
shape_gdf = gpd.read_file(shapef_path)
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

start_year = 1985
end_year = 2019

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

# 1984 - 2021
# CERRA Land, 3x2: sigma = 0.95, LR = 0.0008

# common for all of the SOMs
perc_pca = 0.95

pca = PCA(n_components= perc_pca) # Keeps 95% of the variance
data_train_pca = pca.fit_transform(data_train)
data_test_pca = pca.transform(data_test)
all_data_pca = pca.transform(all_data)
print(data_train_pca.shape)

if start_year == 1985 and end_year== 2019 :
    
    if region_domain == "north":
        
        sigma = 1
        learning_rate = 0.002
        som_col = 2
        som_row = 2
        
        if pr_dataset_str == "ARCIS" :
            folderpath_SOMs = 'your_path/' # your path where you have saved the SOMs
            master_som_name = "ARCIS_north_EU5_filter1000_Z500_mSLP_PCA_95_2by2_LR0.002_sig1_n_1"
            wt_list = [[2, 1], [4, 3]] # ARCIS Central-North
            
        elif pr_dataset_str == "MSWEP" :
            folderpath_SOMs = 'your_path/' # your path where you have saved the SOMs
            master_som_name = "MSWEP_north_EU5_filter1000_Z500_mSLP_PCA_95_2by2_LR0.002_sig1_n_6"
            wt_list = [[2, 4], [1, 3]] # MSWEP Central-North
            
        elif pr_dataset_str == "CERRA_LAND" :
            folderpath_SOMs = 'your_path/' # your path where you have saved the SOMs
            master_som_name = "CERRA_LAND_north_EU5_filter1000_Z500_mSLP_PCA_95_2by2_LR0.002_sig1_n_4"
            wt_list = [[3, 4], [1, 2]] # CERRA Central-North    
        
    elif region_domain == "italy":
        
        sigma = 1
        learning_rate = 0.005
        som_col = 3
        som_row = 2
        
        folderpath_SOMs = 'your_path/' # your path where you have saved the SOMs
        master_som_name = "CERRA_LAND_italy_EU5_filter1000_Z500_mSLP_PCA_95_3by2_LR0.005_sig1_n_3"
        wt_list = [[1, 3, 5], [2, 4, 6]] # CERRA Italy

print("loading " + master_som_name)

som_shape =(som_row, som_col)
min_som = min(som_col, som_row)

number_of_soms = 10
q_win = 100000.
q_error_list = []
t_error_list = []
       

filepath_master_som = glob.glob(folderpath_SOMs + master_som_name + '*')[0]  #this is showing the path and the given file

print(filepath_master_som)
master_som = pickle.load(open(filepath_master_som, 'rb'))
som_weights = master_som._weights
print(som_weights.shape)

# Reverse PCA to return to original features
data_prototypes_pca_inverse = pca.inverse_transform(som_weights)

# new dictionary for the new data
keys = [i for i in product(range(som_row), range(som_col))]  ## DIM OF SOMS
winmap = {key: [] for key in keys}

data_chosen = all_data_pca
test_train_all = "all"

winner_coordinates = np.array([master_som.winner(x) for x in data_chosen]).T
        
# with np.ravel_multi_index we convert the bidimensional
# coordinates to a monodimensional index
cluster_index_list = []
cluster_dates_list = []

cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)
cluster_index_list += [cluster_index] 
dim_som = som_col * som_row

# to get the dates 
cluster_nodes_dates = divide_dates_by_index(time_values, cluster_index, dim_som)

#%%
nx = int(len(lat))
ny = int(len(lon))

# just for graphic reason
#wt_list = [[6, 3, 5], [2, 4, 1]] # CERRA Italy 1984 - 2021

# Correlation 
for i, x in enumerate(data_chosen):
    winmap[master_som.winner(x)].append(i)
    
corr_list = []
for k in range(som_weights.shape[0]):
    for j in range(som_weights.shape[1]):
        index_maps = winmap[(k,j)]
        #print(index_maps) #index of the maps of a single node
        cluster_maps = [data_chosen[i] for i in index_maps] # maps relative to such indices
        
        print("Node " + str(k*som_weights.shape[1] + j+1))
        print(" number of maps: " + str((len(cluster_maps))))
        corr_list_temp = []
        for i in range(len(cluster_maps)):
            corr_list_temp += [np.corrcoef(som_weights[k,j,:], cluster_maps[i])[0,1]]
    
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

#plt.suptitle('SOM Nodes (anom. all data): ' + som_variables_str + ". " + pr_dataset_str + " " + region_domain + " " + str(som_col) + "by" + str(som_row) + " n." + name_best[-2:], x= 0.33 ,fontsize=26)   
plt.savefig(folderpath_SOMs + pr_dataset_str + '_all_anomalyplot_' +master_som_name+'.svg', bbox_inches='tight')
plt.savefig(folderpath_SOMs + pr_dataset_str + '_all_anomalyplot_' +master_som_name+'.png', bbox_inches='tight')
plt.show()

#%% Trend
# Annual time series - Complete with 0
years = range(start_year,end_year+1)
reference_df = pd.DataFrame([year for year in years],  # reference df 
                         columns=['year'])

fig, axs = plt.subplots(nrows=som_row, ncols=som_col,
                        figsize=(24, 10),facecolor='white', dpi=200) 

node = 0
for i in range(som_row):
    for j in range(som_col):
        index_color = node + node*1
        index_color_trend = index_color+1
        date_list = cluster_nodes_dates[node]
        
        df = pd.DataFrame({'date': date_list})  # dataframe with pandas
        df['year'] = df['date'].dt.year         # extract the year 
        
        annual_counts = df.groupby('year').size() # counts
        
        annual_counts_df = annual_counts.reset_index(name='counts')           # turn into df
        
        annual_counts_df_complete= reference_df.merge(annual_counts_df, on=['year'], how='left')
        annual_counts_df_complete['counts'] = annual_counts_df_complete['counts'].fillna(0).astype(int)  # fill the gap years 
        
        years_index = annual_counts_df_complete['year']
        years_counts= annual_counts_df_complete['counts']
        
        m, q, y_fit = linear_trend(years_index - years_index[0], years_counts)
        print("m:", m, "; ", "q:" , q)
        trend, _, p, _, _, _, _, _, _ =  mk.original_test(years_counts)
        print(node+1)
        print(mk.original_test(years_counts))
        
        #axs[i, j].plot(annual_counts.index, annual_counts.values, marker='o', linestyle='-',  linewidth=1)
        #axs[i, j].plot(annual_counts.index, y_fit, label=f'Trend Lineare (y = {m:.2f}x + {q:.2f})', linewidth=2)
        
        axs[i, j].plot(years_index, years_counts, marker='o', linestyle='-', color=colors_trend[index_color], linewidth=1)
        #axs[i, j].plot(years_index, y_fit, label=f'y = {m:.2f}x + {q:.2f}, p:{p:.2f} ', color=colors_trend[index_color_trend], linewidth=2)
        axs[i, j].plot(years_index, y_fit, label=f'y = {m:.2f}x, p:{p:.2f} ', color=colors_trend[index_color_trend], linewidth=2)
        axs[i, j].set_title(f"WT {wt_list[i][j]}", fontsize=26)
        #axs[i, j].set_title(f"N {node}", fontsize=24)
        axs[i, j].set_xlabel('Year', fontsize=21)
        axs[i, j].set_ylabel('Annual frequency', fontsize=21)
        axs[i, j].set_ylim([0,17])
        axs[i, j].grid()
        axs[i, j].tick_params(labelsize=18)
        axs[i, j].legend(fontsize=16)
        node = node + 1
        
fig.tight_layout()
#plt.savefig(folderpath_SOMs + pr_dataset_str + '_annual_trends_' +master_som_name+'.svg', bbox_inches='tight')
#plt.savefig(folderpath_SOMs + pr_dataset_str + '_annual_trends_' +master_som_name+'.png', bbox_inches='tight')
plt.show()

seasons = ["DJF", "MAM", "JJA", "SON"]
reference_df = pd.DataFrame([(year, season) for year in years for season in seasons],  # reference df 
                         columns=['year', 'season'])


# =============================================================================
# fig, axs = plt.subplots(nrows=som_row,ncols=som_col,
#                         figsize=(24, 10),facecolor='white') 
# 
# node = 0
# for i in range(som_row):
#     for j in range(som_col):
#         date_list = cluster_nodes_dates[node]
#         df = pd.DataFrame({'date': date_list})
#         
#         count_per_season = df.groupby([df['date'].dt.year.rename('year'), 
#                            df['date'].apply(get_season).rename('season')]).size()   # season e anno
# 
#         count_per_season_df = count_per_season.reset_index(name='counts')           # turn into df
#         
#         count_per_season_df_complete = reference_df.merge(count_per_season_df, on=['year', 'season'], how='left')
#         count_per_season_df_complete['counts'] = count_per_season_df_complete['counts'].fillna(0).astype(int)  # fill the gap years 
#         
#         for k, season in enumerate(seasons):
#             index_color = k + k*1
#             index_color_trend = index_color+1
#             
#             print(k, season)
#             season_data = count_per_season_df_complete[count_per_season_df_complete['season'] == season]
#             
#             seas_year   = season_data['year']
#             seas_counts = season_data['counts']
#             
#             m, q, y_fit = linear_trend(seas_year, seas_counts)
#             trend, _, p, _, _, _, _, _, _ = mk.original_test(seas_counts.values)
#             print(node+1)
#             print(mk.original_test(seas_counts.values))
#             # Plot
#             axs[i, j].plot(seas_year, seas_counts, marker='o', linestyle='-', color=seasonal_colors[index_color], label=season, alpha=0.8, linewidth=1)
#             axs[i, j].plot(seas_year, y_fit, label=f'y = {m:.2f}x + {q:.2f}, {trend} p:{p:.2f}', color=seasonal_colors[index_color_trend], linewidth=2)
#             
#         axs[i, j].set_title("WT" + str(wt_list[i][j]), fontsize=24)
#         axs[i, j].set_xlabel('Year', fontsize=20)
#         axs[i, j].set_ylabel('Counts', fontsize=20)
#         axs[i, j].grid()
#         axs[i, j].tick_params(labelsize=16)
#         axs[i, j].legend()
#         
#         node = node +1
#         
# fig.tight_layout()
# plt.show()
# 
# =============================================================================
# CERRA only stat. significant WT3 node=1

fig, axs = plt.subplots(nrows=1,ncols=1,
                        figsize=(12, 6),facecolor='white', dpi=200) 

node = 1
date_list = cluster_nodes_dates[node]
df = pd.DataFrame({'date': date_list})

count_per_season = df.groupby([df['date'].dt.year.rename('year'), 
                   df['date'].apply(get_season).rename('season')]).size()   # season e anno

count_per_season_df = count_per_season.reset_index(name='counts')           # turn into df

count_per_season_df_complete = reference_df.merge(count_per_season_df, on=['year', 'season'], how='left')
count_per_season_df_complete['counts'] = count_per_season_df_complete['counts'].fillna(0).astype(int)  # fill the gap years 

for k, season in enumerate(seasons):
    index_color = k + k*1
    index_color_trend = index_color+1
    
    print(k, season)
    season_data = count_per_season_df_complete[count_per_season_df_complete['season'] == season]
    
    seas_year   = season_data['year']
    seas_counts = season_data['counts']
    
    m, q, y_fit = linear_trend(seas_year, seas_counts)
    trend, _, p, _, _, _, _, _, _ = mk.original_test(seas_counts.values)
    print(p)
    if p < 0.10:
        axs.plot(seas_year, seas_counts, marker='o', linestyle='-', color=seasonal_colors[index_color], label=season, linewidth=2)
        axs.plot(seas_year, y_fit, label=f'y = {m:.2f}x, {trend} p:{p:.2f}', color=seasonal_colors[index_color_trend], linewidth=2)
    else :
        axs.plot(seas_year, seas_counts, marker='o', linestyle='-', color=seasonal_colors[index_color], label=season, linewidth=1)
        axs.plot(seas_year, y_fit, label=f'y = {m:.2f}x, {trend} p:{p:.2f}', linestyle='--', color=seasonal_colors[index_color_trend], linewidth=2)
    print(mk.original_test(seas_counts.values))
    # Plot

    
axs.set_title("WT3", fontsize=24)
axs.set_xlabel('Year', fontsize=20)
axs.set_ylabel('Annual frequence', fontsize=20)
axs.grid()
axs.tick_params(labelsize=16)
axs.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 13})

fig.tight_layout()
#plt.savefig(folderpath + pr_dataset_str + '_WT3_annual_trends_' +master_som_name+'.svg', bbox_inches='tight')
#plt.savefig(folderpath + pr_dataset_str + '_WT3_annual_trends_' +master_som_name+'.png', bbox_inches='tight')
plt.show()

#%% COMPOSITES
# Loading ERA5 data
#Z500
name_file = data_path_ERA5 + "ERA5_daily_mean_Geop_500hPa_1985_2019_EU3.nc"     # Z500 (not anomaly)
ds = xr.open_mfdataset(name_file)
ds.sel(lat = slice(60,25), lon = slice(-13,35))                                 # We select a wider domain to show the composites  
#ds = dy.sel(time=extreme_list, method="nearest") 
#print(ds)

# Loading the data into the variables
z_time_values = ds['time'].values
z_values = ds['Z'].values
z_values = z_values / g
z_values = z_values[:,0,:,:]
z_raw = (ds['Z'])/g  
z_lon = ds['lon'].values
z_lat = ds['lat'].values

# MSLP
name_file = data_path_ERA5 + "ERA5_daily_mean_mSLP_sfc_1985_2019_EU3.nc"      # MSLP (not anomaly)
ds = xr.open_mfdataset(name_file)
ds.sel(lat = slice(60,25), lon = slice(-13,35))                                 # We select a wider domain to show the composites  
print(ds)

# Loading the data into the variables
mslp_time_values = ds['time'].values
mslp_values = ds['MSL'].values
mslp_values = mslp_values /100
mslp_raw = (ds['MSL'])/100 #This is the data the NON-anomaly data.
#lon = ds['lon'].values
#lat = ds['lat'].values

## IVT data
name_file = "ERA5_vertical_integral_of_eastward_water_vapour_flux_day_full_sfc_EU_1985_2022.nc"
path_file = data_path_ERA5 + name_file
viwve, time_ERA5, xlat_ERA5, xlong_ERA5, _ = ERA5_IVT(path_file, start_year, end_year, "EU3") # We select a wider domain to show the composites  
viwve = viwve[:,0,:,:]

name_file = "ERA5_vertical_integral_of_northward_water_vapour_flux_day_full_sfc_EU_1985_2022.nc"
path_file = data_path_ERA5 + name_file
viwvn, _, _, _, _ = ERA5_IVT(path_file, start_year, end_year, "EU3")
viwvn = viwvn[:,0,:,:]

## TCWV
name_file = data_path_ERA5 + "ERA5_total_column_water_vapour_day_full_sfc_EU_1985_2022.nc" 
ds = xr.open_mfdataset(name_file, preprocess=select_latlon)
print(ds)

# Loading the data into the variables
TCWV_time_values = ds['time'].values
TCWV_values = ds['TCWV'].values
TCWV_raw = (ds['TCWV']) #This is the data the NON-anomaly data.
TCWV_lon = ds['lon'].values
TCWV_lat = ds['lat'].values
nx = int((TCWV_raw['lat'].size))
ny = int((TCWV_raw['lon'].size))

#%%
# generating single composite
all_data_composites_z500 = single_SOM_map_composite_with_ref_nearest(all_data_pca, time_values, z_values, z_time_values, cluster_nodes_dates)
all_data_composites_IVTE = single_SOM_map_composite_with_ref_nearest(all_data_pca, time_values, viwve, time_ERA5, cluster_nodes_dates)
all_data_composites_IVTN = single_SOM_map_composite_with_ref_nearest(all_data_pca, time_values, viwvn, time_ERA5, cluster_nodes_dates)
all_data_composites_mslp = single_SOM_map_composite_with_ref_nearest(all_data_pca, time_values, mslp_values, mslp_time_values, cluster_nodes_dates)
all_data_composites_TCWV = single_SOM_map_composite_with_ref_nearest(all_data_pca, time_values, TCWV_values, TCWV_time_values, cluster_nodes_dates)

#%%
# Generating the different figures 
# Z500 and mSLP
single_generate_SOM_composites_Z500_mslp(z_lon, z_lat, all_data_composites_z500, all_data_composites_mslp, som_col, som_row,  master_som_name, folderpath_SOMs, flat_wt_list , "all")
single_generate_SOM_composites_Z500_mslp_it(z_lon, z_lat, all_data_composites_z500, all_data_composites_mslp, som_col, som_row,  master_som_name, folderpath_SOMs, flat_wt_list , "all")

# Z500, TCWV, IVT
#single_generate_SOM_composites_Z500_TCWV_IVT(z_lon, z_lat, TCWV_lon, TCWV_lat, data_shaded, data_contour, ivte, ivtn, som_col, som_row, name_som, folderpath, pattern_list, test_or_all=None, chosen_map= "RdYlBu_r"):
single_generate_SOM_composites_Z500_TCWV_IVT(z_lon, z_lat, TCWV_lon, TCWV_lat, xlong_ERA5, xlat_ERA5, all_data_composites_TCWV, all_data_composites_z500, all_data_composites_IVTE, all_data_composites_IVTN, som_col, som_row,  master_som_name,  folderpath_SOMs, flat_wt_list, "all")

#%% SEASONALITY
fig = plt.figure(figsize=(20,13))
#plt.title("Seasonality (all data). SOM " + som_variables_str + " " + pr_dataset_str + " " + region_domain + " " + str(som_col) + "by" + str(som_row) + " Master SOM")

base_index_str = str(som_row) + str(som_col) 

for k in range(len(cluster_nodes_dates)):
    dates = np.array(cluster_nodes_dates[k])
    months = dates.astype('datetime64[M]').astype(int) % 12 + 1
    counts = np.bincount(months)[1:] # [1:] toglie il primo valore relativo al numero 0
    #print(len(counts))
    if len(counts) < 12 :
        N = 12 - len(counts)
        counts = np.pad(counts, (0, N), 'constant')
        print("missing some month")
    #print(len(counts))
    print(counts)
    label_node = "#" + str(k+1)
    
    index_node_str = base_index_str + str(k+1)
    index_node = int(index_node_str)
    print(index_node)
    
    counts_series = pd.Series(counts)
    ax = fig.add_subplot(index_node)
    ax.set_axis_off()
    ax.axis('off')
    #ax.pie(counts, labels=month_str, cmap=seasonal_colormap)
    counts_series.plot.pie(labels=month_str, cmap=seasonal_colormap, autopct=lambda pct: func(pct, counts), counterclock=False, startangle=-270, fontsize=18)

plt.axis('off')
#plt.savefig(folderpath + master_som_name + '_' + '_seasonality_pie_'  + test_or_all + '.png', bbox_inches='tight')
#plt.savefig(folderpath + master_som_name + '_' + '_seasonality_pie_'  + test_or_all + '.svg', bbox_inches='tight')
plt.show()    

#%% PRECIPITATION 
### LOADING THE DATA
## ARCIS
pr_file = path_ARCIS + "/ARCIS3_GG_1985-2019_updated_time.nc"
dy = xr.open_mfdataset(pr_file)
ds = dy.sel(time=extreme_pr_days_list, method="nearest") 
pr_time_ARCIS = ds['time'].values
pr_raw_ARCIS = ds['rr']
pr_ARCIS = ds['rr'].values
print("pr shape ", pr_ARCIS.shape)

pr_lon_ARCIS = ds['lon']
pr_lat_ARCIS = ds['lat']
pr_nx_ARCIS = int((pr_raw_ARCIS['lat'].size))
pr_ny_ARCIS = int((pr_raw_ARCIS['lon'].size))

# PERCENTILE
pr_file_99_ARCIS = path_ARCIS + "/ARCIS3_GG_1985-2019_upd_t_99perc.nc"
dy = xr.open_mfdataset(pr_file_99_ARCIS)
pr_ARCIS_99 = dy['rr'].values[0]
print("pr 99 shape: ", pr_ARCIS_99.shape)

pr_ARCIS_mask = pr_ARCIS_99 / pr_ARCIS_99

#%% MSWEP
pr_file = path_MSWEP + "/All_precip_MSWEP_1985_2019_italy.nc"
dy = xr.open_mfdataset(pr_file)
ds = dy.sel(time=extreme_pr_days_list, method="nearest") 
pr_time_MSWEP = ds['time'].values
pr_raw_MSWEP = ds['precipitation']
pr_MSWEP = ds['precipitation'].values
print("pr shape ", pr_MSWEP.shape)

pr_lon_MSWEP = ds['lon']
pr_lat_MSWEP = ds['lat']
pr_nx_MSWEP = int((pr_raw_MSWEP['lat'].size))
pr_ny_MSWEP = int((pr_raw_MSWEP['lon'].size))

# 99mo percentile
pr_file_99 = path_MSWEP + "/Wet_precip_MSWEP_1985_2019_italy_99percentile.nc"
dy = xr.open_mfdataset(pr_file_99)
pr_MSWEP_99 = dy['precipitation'].values
pr_MSWEP_99 = pr_MSWEP_99[0]
print("pr 99 shape: ", pr_MSWEP_99.shape)

#%% CERRA_Land
pr_file = path_CERRA + "/All_precip_1984_2021_reg6.nc"

dy = xr.open_mfdataset(pr_file)
ds = dy.sel(time=extreme_pr_days_list, method="nearest") 
pr_time_CERRA = ds['time'].values
#pr_time_MSWEP = ds['time'].dt.date.values
#pr_data = ds['precipitation'].values
pr_raw_CERRA = ds['tp']
pr_CERRA = ds['tp'].values
print("pr shape ", pr_CERRA.shape)

pr_lon_CERRA = ds['lon']
pr_lat_CERRA = ds['lat']
pr_nx_CERRA = int((pr_raw_CERRA['lat'].size))
pr_ny_CERRA = int((pr_raw_CERRA['lon'].size))

# sea mask
cerra_mask_file = path_CERRA + "/CERRA_LAND_land_mask_6km.nc"
dy = xr.open_mfdataset(cerra_mask_file)
print(dy)

cerra_mask_lon = dy['lon']
cerra_mask_lat = dy['lat']
cerra_mask_nx = int((cerra_mask_lat.size))
cerra_mask_ny = int((cerra_mask_lon.size))

cerra_mask_values = dy["lsm"].values[0]

cerra_mask_values = np.where(cerra_mask_values==0, np.NaN, cerra_mask_values)

# 99mo percentile
pr_file_99 = path_CERRA + "/Wet_precip_1985_2019_reg6_99percentile.nc"

dy = xr.open_mfdataset(pr_file_99)
pr_CERRA_99 = dy['tp'].values
pr_CERRA_99= pr_CERRA_99[0] * cerra_mask_values
print("pr 99 shape: ", pr_CERRA_99.shape)

#%% CERRA - WARNING AREAS
pr_file = path_CERRA_wa + "/All_precip_CERRA_LAND_1985_2019_italy_wa_mean.nc"

dy = xr.open_mfdataset(pr_file)
ds = dy.sel(time=extreme_pr_days_list, method="nearest") 
pr_time_CERRA = ds['time'].values
pr_raw_CERRA = ds['tp_mean']
pr_CERRA = ds['tp_mean'].values
print("pr shape ", pr_CERRA.shape)

pr_lon_CERRA = ds['lon']
pr_lat_CERRA = ds['lat']
pr_nx_CERRA = int((pr_raw_CERRA['lat'].size))
pr_ny_CERRA = int((pr_raw_CERRA['lon'].size))

# 99mo percentile
pr_file_99 = path_CERRA_wa + "/Wet_precip_CERRA_LAND_1985_2019_italy_wa_mean_99percentile.nc"

dy = xr.open_mfdataset(pr_file_99)
#ds = dy.sel(time=extreme_list, method="nearest") 
pr_CERRA_99 = dy['tp_mean'].values
pr_CERRA_99= pr_CERRA_99[0]
print("pr 99 shape: ", pr_CERRA_99.shape)
#%% COMPOSITES
print("loading pr " + pr_dataset_str)
    
if pr_dataset_str == "ARCIS" :
    pr = pr_ARCIS
    pr_time = pr_time_ARCIS
    pr_99 = pr_ARCIS_99
    pr_mask = pr_ARCIS_mask
    
    pr_lon = pr_lon_ARCIS 
    pr_lat = pr_lat_ARCIS 
    pr_nx  = pr_nx_ARCIS 
    pr_ny  = pr_ny_ARCIS 
 
if pr_dataset_str == "MSWEP" :
    pr = pr_MSWEP
    pr_time = pr_time_MSWEP
    pr_99 = pr_MSWEP_99

    pr_lon = pr_lon_MSWEP 
    pr_lat = pr_lat_MSWEP 
    pr_nx  = pr_nx_MSWEP 
    pr_ny  = pr_ny_MSWEP 
    
if pr_dataset_str == "CERRA_LAND" :
    pr = pr_CERRA
    pr_time = pr_time_CERRA
    pr_99 = pr_CERRA_99
    #pr_mask = cerra_mask_values
    
    pr_lon = pr_lon_CERRA 
    pr_lat = pr_lat_CERRA 
    pr_nx  = pr_nx_CERRA 
    pr_ny  = pr_ny_CERRA 



nx = len(pr[0])
print(nx)
ny = len(pr[0][0])
print(ny)
            
maps_prob_perc = [None for _ in range(len(cluster_nodes_dates)) ]

list_map_greater_temp = []
for j in range(len(cluster_nodes_dates)): # iterates over all nodes 
    list_map_temp = []
    print("node", j+1)
    for k in range(len(cluster_nodes_dates[j])): # Iterates over all maps within each node
        

        indx = np.where(pr_time[:].astype('datetime64[D]')==cluster_nodes_dates[j][k].astype('datetime64[D]'))[0]
        #indx = np.where(time_values[:]==cluster_dates_list[i][j][k], "nearest")[0][0]
        #print(cluster_dates_list[0][0][k])
        #print(time_values[indx])
        #print("")
        list_map_temp += [pr[indx]] 
        #print(pr_time[indx])
        #print(cluster_nodes_dates[j][k])
        
    maps_array = np.array(list_map_temp) # all the maps that are associated with a node
    #print(maps_array.shape)
    
    if maps_array.shape == (0,) :    
        # sometimes you do not have maps that satisfy the condition, so you have 0 maps
        # in this case I put the map as 0
        print("no maps")
        prob_map = [ [0 for i in range(len(pr[0][0])) ] for j in range(len(pr[0]))]
        
    else:
                      
        maps_array = maps_array[:,0,:,:]
        #print(maps_array.shape)
        #print(pr_99.shape) # check if they  have the same dimension

        list_map_greater = np.sum(maps_array >= pr_99, axis=0)  # del singolo nodo
        list_map_greater_temp += [list_map_greater] #
            # somma tutte le volte che è soddisfatta la condizione maps_array >= pr_99 per ogni singola mappa presente in quel nodo
  
        #print(list_map_greater.shape)
        prob_map = list_map_greater / maps_array.shape[0] # divido per il numero di giorni presenti nella mappa
        print("number of maps", maps_array.shape[0])
        #maps_composite[i][j] = list_map_mean[0,:,:]
        #print(prob_map.shape)
 
    maps_prob_perc[j]=prob_map # reshape

list_map_greater_total = np.sum(list_map_greater_temp, axis=0)

maps_prob_perc_2 = [None for _ in range(len(cluster_nodes_dates)) ]

for j in range(len(cluster_nodes_dates)): # iterates over all nodes 
    
    #print(list_map_greater.shape)
    prob_map = list_map_greater_temp[j] / list_map_greater_total # divido per il numero di giorni presenti nella mappa
    #maps_composite[i][j] = list_map_mean[0,:,:]
    #print(prob_map.shape)
 
    maps_prob_perc_2[j]=prob_map # reshape

all_data_prob_maps = maps_prob_perc_2

datacrs = ccrs.PlateCarree()

#You will set this to the dimensions of the SOM.  ## DIM OF SOMS
fig, axs = plt.subplots(nrows=som_row,ncols=som_col, 
                        subplot_kw={'projection': ccrs.PlateCarree()},
                        #figsize=(30, 20),facecolor='white') #Italy NI 3X3
                        figsize=(38,20),facecolor='white') #EU AND NI 2X3
axs=axs.flatten()
for k in range(len(maps_prob_perc)):
    # to check
    #lev_start = SOM_mslp.min()
    #print(lev_start)             
    #lev_stop = SOM_mslp.max()
    #print(lev_stop)
    
    levs = np.arange(0, 69, 4) 
    #levs = np.arange(0, 18, 1) 
    n_cmap = len(levs)
    norm = mpl.colors.BoundaryNorm(levs, n_cmap)
    cmap = plt.cm.get_cmap(prob_colormap, n_cmap)
    #cs2=axs[k].contourf(pr_lon, pr_lat, all_data_prob_maps[k]*100*pr_mask,
    cs2=axs[k].contourf(pr_lon, pr_lat, all_data_prob_maps[k]*100,
                      n_cmap, cmap=cmap, norm=norm, levels=levs, transform = ccrs.PlateCarree(), extend='both')               
        
    
    axs[k].set_extent([5, 19, 48, 36], ccrs.PlateCarree()) # all Italy
    domain = "IT"
    #axs[k].set_extent([6.5, 14.2, 47.5, 41.9], ccrs.PlateCarree()) # ARCIS domain
    #domain = "NI"
    
    
    axs[k].coastlines()
    axs[k].add_feature(cfeature.BORDERS) 
    #axs[(k*4)+i].scatter(-156.36,71.19, c='yellow',marker= 'o',s=120, linewidth=2,edgecolors= "black" ,zorder= 4,transform=datacrs)
    
    
    # Title each subplot 
    #axs[k].set_title('Node:'+str(k+1) + " Freq:" + str(frequencies[k,i]), fontsize=18)
    axs[k].set_title('WT'+str(flat_wt_list[k]), fontsize=28)
    
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.25, top=0.9, left=0.05, right=0.6,
                wspace=0.05, hspace=0.25)
        
cbar_ax = fig.add_axes([0.08, 0.2, 0.5, 0.02])
#cbar=fig.colorbar(cs2,cax=cbar_ax, ticks = np.arange(lev_start, np.abs(lev_start)+lev_step, lev_step*2),orientation='horizontal')
#cbar=fig.colorbar(cs2,cax=cbar_ax, ticks = levs,orientation='horizontal')
cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, ticks=levs,
                             spacing='uniform', format='%1i',  orientation='horizontal')
cbar.ax.tick_params(labelsize=22)
cbar.set_label("% of extreme events belonging to each WT", fontsize=26)

#plt.suptitle('Probability above 99th per. (' + pr_dataset_str + ") " + region_domain + " " + str(som_col) + "by" + str(som_row) + " n." + master_som_name[-2:], x= 0.33 ,fontsize=26)   
#plt.savefig(folderpath + master_som_name + '_' + 'Prob_maps_'  + test_or_all + '_wet.png', bbox_inches='tight')
#plt.savefig(folderpath + master_som_name + '_' + 'Prob_maps_'  +test_or_all + '_wet.svg', bbox_inches='tight')
plt.show()

