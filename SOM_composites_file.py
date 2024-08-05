# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 19:41:28 2024

@author: Criss
"""

import os
#os.environ["OMP_NUM_THREADS"] = '1' # this is necessary to avoid memory leak

os.chdir('C:\\Users\\Criss\\Documents\\Lavoro\\Assegno_2024_2025\\Codici')

# this code generates the composites and maps
import importlib
from functions_SOM import *
from SOM_data_file import * 

import SOM_variable_file
importlib.reload(SOM_variable_file)
#from SOM_variable_file import * 
globals().update(vars(SOM_variable_file))
#%% Loading the SOM
# CHOOSE THIS FROM THE OTHER FILE
#som_variables_str = "Z500_mSLP"
#som_date = "2024-07-22"

#som_variables_str = "pr"
#som_date = "2024-06-27"

som_variables_str = "Z500_pr"
som_date = "2024-07-04"

som_col = 5
som_row = 5

norm_factor = norm_factor_CERRA_LAND

som_shape =(som_row, som_col)
min_som = min(som_col, som_row)
#x= 3 #columns
#y= 4 #row
#input_length = len_datatrain #This is value is the the length of the latitude X longitude. It is the second value in the data_train.shape step. 
sigma = min_som -1        #The sigma value must be y-1. 
#sigma = 
learning_rate = 0.0005  #Learning Rate 
qerror_list = []
q_win = 100000.
number_of_soms = 10

#"Z500_mSLP"
#som_names = dataset_str + "_" + domain_region + "_" + EU_domain + "_45rm_" + som_variables_str + "_" + som_date + "_" + str(som_col) + "by" + str(som_row) + "_LR" + str(learning_rate) + "_sig" + str(sigma) + "_n_"

#pr and Z500_pr
som_names = dataset_str + "_" + domain_region + "_" + EU_domain + "_45rm_" + som_variables_str + "_nomask_" + som_date + "_" + str(som_col) + "by" + str(som_row) + "_LR" + str(learning_rate) + "_sig" + str(sigma) + "_n_"

print(som_names)

names = ([os.path.splitext(os.path.split(x)[-1])[0] for x in glob.glob(folderpath + som_names + '*')]) #this might be different for you

#but this is just grabbing the first few characters of my names of my file (see above how I named them, for example som_8)

filepaths = glob.glob(folderpath + som_names + '*')  #this is showing the path and the given file

print(names)

#%%
print("loading " + dataset_str + '_extreme_dates_' + domain_region + '.npy')
#extreme_dates = np.load(PATH + dataset_str + '_extreme_dates.npy', allow_pickle=True) #extremes 
#extreme_dates = np.load(PATH + dataset_str + '_extreme_dates_ITALY.npy', allow_pickle=True) #extremes 
extreme_dates = np.load(PATH + dataset_str + '_extreme_dates_' + domain_region + '.npy', allow_pickle=True) #extremes 

print(len(extreme_dates))

#%%
#som_variables_str = "Z500_mSLP"

print("ATTENZIONE Stai caricando i dati di "+ domain_region + " " + EU_domain + " SOM di " + som_variables_str)
data_train = np.load(PATH + dataset_str +  '_' + domain_region + '_' + som_variables_str + '_som_data_train_anomalies_' + EU_domain + '_45rm.npy')
data_test =  np.load(PATH + dataset_str+ '_' + domain_region + '_' + som_variables_str + '_som_data_test_anomalies_' + EU_domain + '_45rm.npy')
all_data = np.load(PATH + dataset_str + '_' + domain_region + '_' + som_variables_str + '_som_all_data_anomalies_' + EU_domain + '_45rm.npy')
time_values = np.load(PATH +dataset_str + '_' + domain_region + '_' + som_variables_str + '_som_time_data_anomalies_' + EU_domain + '_45rm.npy')
z_raw = xr.open_dataset(PATH + dataset_str + '_' + domain_region + '_' + som_variables_str + '_som_raw_anomalies_' + EU_domain + '_45rm.nc')

lon = z_raw['lon'].values
lat = z_raw['lat'].values
nx = int((z_raw['lat'].size))
ny = int((z_raw['lon'].size))
#ndays =int((z_raw['time'].size)) #non viene usato mai

print(data_train.shape)
len_datatrain = len(data_train[0])

time_values_test = time_values[len(data_train):]

#%%
#som_variables_str = "pr" 

print("ATTENZIONE Stai caricando i dati di "+ domain_region + " " + EU_domain + " SOM di " + som_variables_str)
data_train = np.load(PATH + dataset_str +  '_' + domain_region + '_' + som_variables_str + '_som_data_train_anomalies_' + EU_domain + '_45rm_nomask.npy')
data_test =  np.load(PATH + dataset_str+ '_' + domain_region + '_' + som_variables_str + '_som_data_test_anomalies_' + EU_domain + '_45rm_nomask.npy')
all_data = np.load(PATH + dataset_str + '_' + domain_region + '_' + som_variables_str + '_som_all_data_anomalies_' + EU_domain + '_45rm_nomask.npy')
time_values = np.load(PATH +dataset_str + '_' + domain_region + '_' + som_variables_str + '_som_time_data_anomalies_' + EU_domain + '_45rm_nomask.npy')
z_raw = xr.open_dataset(PATH + dataset_str + '_' + domain_region + '_' + som_variables_str + '_som_raw_anomalies_' + EU_domain + '_45rm_nomask.nc')

lon = z_raw['lon'].values
lat = z_raw['lat'].values
nx = int((z_raw['lat'].size))
ny = int((z_raw['lon'].size))
#ndays =int((z_raw['time'].size)) #non viene usato mai

print(data_train.shape)
len_datatrain = len(data_train[0])

time_values_test = time_values[len(data_train):]


#%%
#som_variables_str = "Z500_pr"

print("ATTENZIONE Stai caricando i dati di "+ domain_region + " " + EU_domain + " SOM di " + som_variables_str)
data_train = np.load(PATH + dataset_str +  '_' + domain_region + '_' + som_variables_str + '_som_data_train_anomalies_' + EU_domain + '_45rm.npy')
data_test =  np.load(PATH + dataset_str+ '_' + domain_region + '_' + som_variables_str + '_som_data_test_anomalies_' + EU_domain + '_45rm.npy')
all_data = np.load(PATH + dataset_str + '_' + domain_region + '_' + som_variables_str + '_som_all_data_anomalies_' + EU_domain + '_45rm.npy')
time_values = np.load(PATH +dataset_str + '_' + domain_region + '_' + som_variables_str + '_som_time_data_anomalies_' + EU_domain + '_45rm.npy')
z_raw = xr.open_dataset(PATH + dataset_str + '_' + domain_region + '_' + som_variables_str + '_Z500_raw_' + EU_domain + '_45rm.nc')
pr_raw = xr.open_dataset(PATH + dataset_str + '_' + domain_region + '_' + som_variables_str + '_pr_raw_' + EU_domain + '_45rm.nc')

lon = z_raw['lon'].values
lat = z_raw['lat'].values
nx = int((z_raw['lat'].size))
ny = int((z_raw['lon'].size))

#pr_lon = pr_raw['lon'].values
#pr_lat = pr_raw['lat'].values
#pr_nx = int((pr_raw['lat'].size))
#pr_ny = int((pr_raw['lon'].size))

print(nx, ny)
print(pr_nx, pr_ny)
#ndays =int((z_raw['time'].size)) #non viene usato mai

print(data_train.shape)
len_datatrain = len(data_train[0])

time_values_test = time_values[len(data_train):]

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
print(filepaths)
print(names)


#%%
all_data_composites_z500 = SOM_maps_composites_with_ref_nearest(all_data, time_values, z_values, z_time_values, som_shape, number_of_soms,  filepaths, names, som_col, som_row)
#all_data_composites_IVTE = SOM_maps_composites_with_ref_nearest(all_data, time_values, viwve, time_ERA5, som_shape, number_of_soms,  filepaths, names, som_col, som_row)
#all_data_composites_IVTN = SOM_maps_composites_with_ref_nearest(all_data, time_values, viwvn, time_ERA5, som_shape, number_of_soms,  filepaths, names, som_col, som_row)
all_data_composites_mslp = SOM_maps_composites_with_ref_nearest(all_data, time_values, mslp_values, mslp_time_values, som_shape, number_of_soms,  filepaths, names, som_col, som_row)

#%%
generate_SOM_composites_Z500_mslp(all_data_composites_z500, all_data_composites_mslp, names, som_col, som_row, z_lon, z_lat, som_variables_str, "sat")

#%%
# Vogliamo creare la mappa con la precipitazione massima

def SOM_maps_pr_max(data_values_som, time_values_som, data_values, time_values, som_shape, number_soms, filepath_som, som_names, som_col, som_row):
    """
    Generate composites of data variable for the differents nodes of the differents soms

    Parameters
    ----------
    data_values_som : .npy
        Values of train data or test data.
    time_values_som : .npy
        Time values of train data or test data.
    data_values : TYPE
        Precipitation values
    time_values : TYPE
        Precipitation time values
    som_shape : TYPE
        Shape of SOM ex (2,2).
    number_soms : int
        Number of Soms, usually 10.
    filepath_som : string
        Path where are stored the SOMs.
    som_names : list of strings
        name of soms.

    Returns
    -------
    maps_composite : list of maps
        maps value of composites of given variable.

    """
    # generate cluster index list and dates dalla variabile di riferimento
    cluster_index_list = []
    cluster_dates_list = []
    for path, name in zip(filepath_som, som_names):
        with open (path, 'rb') as f:
            file = pickle.load(f) #This is loading every single som in that location
            #print("ok1")
    # each neuron represents a cluster
            winner_coordinates = np.array([file.winner(x) for x in data_values_som]).T
            #print("ok2")
    # with np.ravel_multi_index we convert the bidimensional
    # coordinates to a monodimensional index
            cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)
            #print("ok3")
            cluster_index_list += [cluster_index] 
            #print("ok4")
            dim_som = som_col * som_row
            #print("ok5")
            cluster_nodes_dates = divide_dates_by_index(time_values_som, cluster_index, dim_som)
            cluster_dates_list += [cluster_nodes_dates]
    
    print("ok_for")
    
    # NEW LINEEE
    nx = len(data_values[0])
    print(nx)
    ny = len(data_values[0][0])
    print(ny)
            
    maps_pr_max = [ [None for _ in range(len(cluster_dates_list[0])) ] for i in range(number_soms)]
    for i in range(len(cluster_dates_list)):
        for j in range(len(cluster_dates_list[i])):
            list_map_temp = []
            for k in range(len(cluster_dates_list[i][j])):
                indx = np.where(time_values[:].astype('datetime64[D]')==cluster_dates_list[i][j][k].astype('datetime64[D]'))[0]
                #indx = np.where(time_values[:]==cluster_dates_list[i][j][k], "nearest")[0][0]
                
                #print(cluster_dates_list[0][0][k])
                #print(time_values[indx])
                #print("")
                list_map_temp += [data_values[indx]] 
                print(time_values[indx])
                print(cluster_dates_list[i][j][k])
                
            list_map_max = np.max(list_map_temp, axis=0)  # computes the maximum values 
            print("shape")
            print(list_map_max.shape)
            #maps_composite[i][j] = list_map_mean[0,:,:]
           
            try:
                maps_pr_max[i][j]=np.reshape(list_map_max,(nx,ny)) # reshape
            except ValueError:
                maps_pr_max[i][j]=np.zeros((nx,ny))
            #print(len(list_map_temp))
    
    return maps_pr_max

pr = pr_CERRA
pr_time = pr_time_CERRA

all_data_composites_pr = SOM_maps_pr_max(all_data, time_values, pr, pr_time, som_shape, number_of_soms,  filepaths, names, som_col, som_row)
#%%
def generate_SOM_pr_max_IT(data, names_som, lon, lat, dataset_name = None, test_or_all=None):
    
    datacrs = ccrs.PlateCarree()
    for i in range(len(data)):    
        """
        #You will set this to the dimensions of the SOM.  ## DIM OF SOMS
        fig, axs = plt.subplots(nrows=som_row,ncols=som_col, 
                                subplot_kw={'projection':ccrs.LambertConformal(central_longitude=lon.mean(), central_latitude=lat.mean(), standard_parallels=(30, 60))},
                                figsize=(30, 15),facecolor='white') 
        """
        #You will set this to the dimensions of the SOM.  ## DIM OF SOMS
        fig, axs = plt.subplots(nrows=som_row,ncols=som_col, 
                                subplot_kw={'projection': ccrs.PlateCarree()},
                                #figsize=(30, 20),facecolor='white') #Italy NI 3X3
                                figsize=(30,20),facecolor='white') #EU AND NI 2X3
        axs=axs.flatten()
        for k in range(len(data[i])):
            # to check
            #lev_start = SOM_mslp.min()
            #print(lev_start)             
            #lev_stop = SOM_mslp.max()
            #print(lev_stop)
        
            #levs = np.arange(0, 400, 15)
            levs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 250, 300, 350, 400, 450]
            #levs = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 180, 200, 250, 300]
            #levs = np.arange(0, 126, 5) v1
            # THIS IS SATURATED
            #levs = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 100]
            n_cmap = len(levs)
            norm = mpl.colors.BoundaryNorm(levs, n_cmap)
            cmap = plt.cm.get_cmap(precip_colormap25, n_cmap)
            cs2=axs[k].contourf(lon, lat, data[i][k],
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
            axs[k].set_title('Node:'+str(k+1), fontsize=18)

            plt.tight_layout()
            fig.subplots_adjust(bottom=0.25, top=0.9, left=0.05, right=0.6,
                        wspace=0.05, hspace=0.25)
        
        cbar_ax = fig.add_axes([0.08, 0.2, 0.5, 0.02])
        #cbar=fig.colorbar(cs2,cax=cbar_ax, ticks = np.arange(lev_start, np.abs(lev_start)+lev_step, lev_step*2),orientation='horizontal')
        #cbar=fig.colorbar(cs2,cax=cbar_ax, ticks = levs,orientation='horizontal')
        cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, ticks=levs,
                                     spacing='uniform', format='%1i',  orientation='horizontal')
        
        cbar.set_label("Pr (mm)", fontsize=22)
        
        plt.suptitle('Pr Max (' + dataset_name + " - " + test_or_all + " data) "+ "SOM (" + som_variables_str + ") n." + names_som[i][-2:] , x= 0.33 ,fontsize=28)   
        plt.savefig(folderpath + 'Pr_max_' + dataset_name + "_" +names_som[i]+ '_' + test_or_all + '_' + domain +'_sat.png', bbox_inches='tight')
        plt.show()


generate_SOM_pr_max_IT(all_data_composites_pr*sea_mask, names, pr_lon, pr_lat, dataset_str, "all")

#%%
all_data_composites_pr = SOM_maps_composites_with_ref_nearest(all_data, time_values, pr, pr_time, som_shape, number_of_soms,  filepaths, names, som_col, som_row)
#%%
def generate_SOM_composites_pr_IT(data, names_som, lon, lat, som_col, som_row, dataset_name = None, test_or_all=None):
    
    datacrs = ccrs.PlateCarree()
    for i in range(len(data)):    
        """
        #You will set this to the dimensions of the SOM.  ## DIM OF SOMS
        fig, axs = plt.subplots(nrows=som_row,ncols=som_col, 
                                subplot_kw={'projection':ccrs.LambertConformal(central_longitude=lon.mean(), central_latitude=lat.mean(), standard_parallels=(30, 60))},
                                figsize=(30, 15),facecolor='white') 
        """
        #You will set this to the dimensions of the SOM.  ## DIM OF SOMS
        fig, axs = plt.subplots(nrows=som_row,ncols=som_col, 
                                subplot_kw={'projection': ccrs.PlateCarree()},
                                #figsize=(30, 20),facecolor='white') #Italy NI 3X3
                                figsize=(30,20),facecolor='white') #EU AND NI 2X3
        axs=axs.flatten()
        for k in range(len(data[i])):
            # to check
            #lev_start = SOM_mslp.min()
            #print(lev_start)             
            #lev_stop = SOM_mslp.max()
            #print(lev_stop)
        
            #levs = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 125, 150, 200, 250, 300, 350, 400]
            #levs = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 180, 200]
            #levs = np.arange(0, 126, 5) v1
            # THIS IS SATURATED
            #levs = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 100]
            #levs = [ 0,  1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
            levs = np.arange(0, 105, 5)
            n_cmap = len(levs)
            norm = mpl.colors.BoundaryNorm(levs, n_cmap)
            cmap = plt.cm.get_cmap(precip_colormap25, n_cmap)
            cs2=axs[k].contourf(lon, lat, data[i][k],
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
            axs[k].set_title('Node:'+str(k+1), fontsize=18)

            plt.tight_layout()
            fig.subplots_adjust(bottom=0.25, top=0.9, left=0.05, right=0.6,
                        wspace=0.05, hspace=0.25)
        
        cbar_ax = fig.add_axes([0.08, 0.2, 0.5, 0.02])
        #cbar=fig.colorbar(cs2,cax=cbar_ax, ticks = np.arange(lev_start, np.abs(lev_start)+lev_step, lev_step*2),orientation='horizontal')
        #cbar=fig.colorbar(cs2,cax=cbar_ax, ticks = levs,orientation='horizontal')
        #cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, ticks=levs,
                                     #spacing='uniform', format='%1i',  orientation='horizontal')
        cbar=fig.colorbar(cs2,cax=cbar_ax, ticks = levs,orientation='horizontal')
        
        cbar.set_label("Pr (mm)", fontsize=22)
        
        plt.suptitle('Composites of pr (' + dataset_name + " - " + test_or_all + " data) "+ "SOM (" + som_variables_str + ") n." + names_som[i][-2:] , x= 0.33 ,fontsize=28)   
        plt.savefig(folderpath + 'Composites_pr_' + dataset_name + "_" +names_som[i]+ '_' + test_or_all + '_' + domain +'_sat.png', bbox_inches='tight')
        plt.show()
generate_SOM_composites_pr_IT(all_data_composites_pr*sea_mask, names, pr_lon, pr_lat, som_col, som_row, dataset_str, "all")

#%%
# I want the map of the 99th percentile
#path_CERRA = "C:/Users/Criss/Documents/Lavoro/Assegno_2024_2025/Dati/CERRA_LAND"            # path 
#pr_file = path_MSWEP + "/MSWEP_1991_2020_Italy.nc"

pr_file_99 = path_CERRA + "/All_precip_CERRA_LAND_1985_2019_italy_99_percentile.nc"

dy = xr.open_mfdataset(pr_file_99)
#ds = dy.sel(time=extreme_list, method="nearest") 
pr_CERRA_99 = dy['tp'].values

pr_CERRA_99= pr_CERRA_99[0] * prova_mask[0]
#%%
#pr_lon = ds['lon']
#pr_lat = ds['lat']

fig = plt.figure(figsize=(12, 8))    
fig.suptitle("99th percentile", fontsize=20)

# Figure - base
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
#levs = [ 0,  1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
#levs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 250, 300, 350, 400, 450]
#levs = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 180, 200, 250, 300]
levs = np.arange(0, 105, 5) 
# THIS IS SATURATED
#levs = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 100]
n_cmap = len(levs)
norm = mpl.colors.BoundaryNorm(levs, n_cmap)
cmap = plt.cm.get_cmap(precip_colormap25, n_cmap)
cs2=ax.contourf(pr_lon, pr_lat, pr_CERRA_99,
                  n_cmap, cmap=cmap, norm=norm, levels=levs, transform = ccrs.PlateCarree(), extend='both')               
ax.add_feature(cfeature.BORDERS)   
ax.add_feature(cfeature.COASTLINE)
ax.set_extent([5, 19, 48, 36], ccrs.PlateCarree()) # all Italy

cbar_ax = fig.add_axes([0.85, 0.1, 0.02, 0.8])
#cbar=fig.colorbar(cs2,cax=cbar_ax, ticks = np.arange(lev_start, np.abs(lev_start)+lev_step, lev_step*2),orientation='horizontal')
#cbar=fig.colorbar(cs2,cax=cbar_ax, ticks = levs,orientation='horizontal')
cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, ticks=levs,
                             spacing='uniform', format='%1i',  orientation='vertical')

cbar.set_label("Pr (mm)", fontsize=22)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False, linewidth=0.33, color='k', alpha=0.5)
gl.right_labels = gl.top_labels = False
plt.savefig(folderpath + "Pr_99_percentile_CERRA_LAND.png")
#fig.tight_layout()

#%%
# I want to calculate the probability of being above the 99mo percentile



#%%
pr = pr_CERRA
pr_time = pr_time_CERRA

def SOM_maps_prob_above_perc(data_values_som, time_values_som, data_values, time_values, map_perc, som_shape, number_soms, filepath_som, som_names, som_col, som_row):
    """
    Generate composites of data variable for the differents nodes of the differents soms

    Parameters
    ----------
    data_values_som : .npy
        Values of train data or test data.
    time_values_som : .npy
        Time values of train data or test data.
    data_values : TYPE
        Precipitation values
    time_values : TYPE
        Precipitation time values
    som_shape : TYPE
        Shape of SOM ex (2,2).
    number_soms : int
        Number of Soms, usually 10.
    filepath_som : string
        Path where are stored the SOMs.
    som_names : list of strings
        name of soms.

    Returns
    -------
    maps_composite : list of maps
        maps value of composites of given variable.

    """
    # generate cluster index list and dates dalla variabile di riferimento
    cluster_index_list = []
    cluster_dates_list = []
    for path, name in zip(filepath_som, som_names):
        with open (path, 'rb') as f:
            file = pickle.load(f) #This is loading every single som in that location
            #print("ok1")
    # each neuron represents a cluster
            winner_coordinates = np.array([file.winner(x) for x in data_values_som]).T
            #print("ok2")
    # with np.ravel_multi_index we convert the bidimensional
    # coordinates to a monodimensional index
            cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)
            #print("ok3")
            cluster_index_list += [cluster_index] 
            #print("ok4")
            dim_som = som_col * som_row
            #print("ok5")
            cluster_nodes_dates = divide_dates_by_index(time_values_som, cluster_index, dim_som)
            cluster_dates_list += [cluster_nodes_dates]
    
    print("ok_for")
    
    # NEW LINEEE
    nx = len(data_values[0])
    print(nx)
    ny = len(data_values[0][0])
    print(ny)
            
    maps_prob_perc = [ [None for _ in range(len(cluster_dates_list[0])) ] for i in range(number_soms)]
    for i in range(len(cluster_dates_list)):
        for j in range(len(cluster_dates_list[i])):
            list_map_temp = []
            for k in range(len(cluster_dates_list[i][j])):
                indx = np.where(time_values[:].astype('datetime64[D]')==cluster_dates_list[i][j][k].astype('datetime64[D]'))[0]
                #indx = np.where(time_values[:]==cluster_dates_list[i][j][k], "nearest")[0][0]
                
                #print(cluster_dates_list[0][0][k])
                #print(time_values[indx])
                #print("")
                list_map_temp += [data_values[indx]] 
                print(time_values[indx])
                print(cluster_dates_list[i][j][k])
                
            maps_array = np.array(list_map_temp)
            print(maps_array.shape)
            maps_array = maps_array[:,0,:,:]
            print(maps_array.shape)
            print(map_perc.shape)
            list_map_greater = np.sum(maps_array >= map_perc, axis=0)  # computes the maximum values 
            print(list_map_greater.shape)
            prob_map = list_map_greater / maps_array.shape[0]
            print(maps_array.shape[0])
            #maps_composite[i][j] = list_map_mean[0,:,:]
            print(prob_map.shape)
            maps_prob_perc[i][j]=prob_map # reshape

    
    return maps_prob_perc


all_data_prob_maps = SOM_maps_prob_above_perc(all_data, time_values, pr, pr_time, pr_CERRA_99, som_shape, number_of_soms,  filepaths, names, som_col, som_row)



def generate_SOM_maps_prob(data, names_som, lon, lat, som_col, som_row, dataset_name = None, test_or_all=None):
    
    datacrs = ccrs.PlateCarree()
    for i in range(len(data)):    
        """
        #You will set this to the dimensions of the SOM.  ## DIM OF SOMS
        fig, axs = plt.subplots(nrows=som_row,ncols=som_col, 
                                subplot_kw={'projection':ccrs.LambertConformal(central_longitude=lon.mean(), central_latitude=lat.mean(), standard_parallels=(30, 60))},
                                figsize=(30, 15),facecolor='white') 
        """
        #You will set this to the dimensions of the SOM.  ## DIM OF SOMS
        fig, axs = plt.subplots(nrows=som_row,ncols=som_col, 
                                subplot_kw={'projection': ccrs.PlateCarree()},
                                #figsize=(30, 20),facecolor='white') #Italy NI 3X3
                                figsize=(30,20),facecolor='white') #EU AND NI 2X3
        axs=axs.flatten()
        for k in range(len(data[i])):
            # to check
            #lev_start = SOM_mslp.min()
            #print(lev_start)             
            #lev_stop = SOM_mslp.max()
            #print(lev_stop)
        
            levs = np.arange(0, 61, 3) 
            n_cmap = len(levs)
            norm = mpl.colors.BoundaryNorm(levs, n_cmap)
            cmap = plt.cm.get_cmap(precip_colormap25, n_cmap)
            cs2=axs[k].contourf(lon, lat, data[i][k]*100,
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
            axs[k].set_title('Node:'+str(k+1), fontsize=18)

            plt.tight_layout()
            fig.subplots_adjust(bottom=0.25, top=0.9, left=0.05, right=0.6,
                        wspace=0.05, hspace=0.25)
        
        cbar_ax = fig.add_axes([0.08, 0.2, 0.5, 0.02])
        #cbar=fig.colorbar(cs2,cax=cbar_ax, ticks = np.arange(lev_start, np.abs(lev_start)+lev_step, lev_step*2),orientation='horizontal')
        #cbar=fig.colorbar(cs2,cax=cbar_ax, ticks = levs,orientation='horizontal')
        cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, ticks=levs,
                                     spacing='uniform', format='%1i',  orientation='horizontal')
        
        cbar.set_label("Perc. prob. [%]", fontsize=22)
        
        plt.suptitle('Probability above 99th per. (' + dataset_name + " - " + test_or_all + " data) "+ "SOM (" + som_variables_str + ") n." + names_som[i][-2:] , x= 0.33 ,fontsize=28)   
        plt.savefig(folderpath + 'Prob_maps_' + dataset_name + "_" +names_som[i]+ '_' + test_or_all + '_' + domain +'_sat2.png', bbox_inches='tight')
        plt.show()

generate_SOM_maps_prob(all_data_prob_maps*sea_mask, names, pr_lon, pr_lat, som_col, som_row, dataset_str, "all")



#%%
fig = plt.figure(figsize=(12, 8))    
fig.suptitle("Probabilità", fontsize=20)

# Figure - base
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
#levs = [ 0,  1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
#levs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 250, 300, 350, 400, 450]
#levs = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 180, 200, 250, 300]
levs = np.arange(0, 105, 5) 
# THIS IS SATURATED
#levs = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 100]
n_cmap = len(levs)
norm = mpl.colors.BoundaryNorm(levs, n_cmap)
cmap = plt.cm.get_cmap(precip_colormap, n_cmap)
cs2=ax.contourf(pr_lon, pr_lat, prova_prob[0][0]*prova_mask[0]*100,
                  n_cmap, cmap=cmap, norm=norm, levels=levs, transform = ccrs.PlateCarree(), extend='both')               
    
ax.add_feature(cfeature.COASTLINE)
ax.set_extent([5, 19, 48, 36], ccrs.PlateCarree()) # all Italy

cbar_ax = fig.add_axes([0.85, 0.1, 0.02, 0.8])
#cbar=fig.colorbar(cs2,cax=cbar_ax, ticks = np.arange(lev_start, np.abs(lev_start)+lev_step, lev_step*2),orientation='horizontal')
#cbar=fig.colorbar(cs2,cax=cbar_ax, ticks = levs,orientation='horizontal')
cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, ticks=levs,
                             spacing='uniform', format='%1.f',  orientation='vertical')

cbar.set_label("%", fontsize=22)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False, linewidth=0.33, color='k', alpha=0.5)
gl.right_labels = gl.top_labels = False




#%%

