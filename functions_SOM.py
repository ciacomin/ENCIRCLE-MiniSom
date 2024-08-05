# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 15:28:43 2024

@author: Criss
"""
import importlib
# this code has all the functions needed 
import numpy as np
import pandas as pd
#import glob
import xarray as xr
import warnings
import minisom
import pickle
from minisom import asymptotic_decay
import cartopy.mpl.ticker as cticker
from cartopy.util import add_cyclic_point
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from numpy import savetxt
from numpy import loadtxt
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob
import matplotlib.pyplot as plt
import os
from itertools import product
import winsound
from datetime import date
#from xarray import DataArray

from functions_new_version import *


import SOM_variable_file
importlib.reload(SOM_variable_file)
#from SOM_variable_file import * 
globals().update(vars(SOM_variable_file))


def SOM_maps_composites_with_ref(data_values_som, time_values_som, data_values, time_values, som_shape, number_soms, filepath_som, som_names):
    """
    Generate composites of data variable for the differents nodes of the differents soms

    Parameters
    ----------
    data_values_som : .npy
        Values of train data or test data.
    time_values_som : .npy
        Time values of train data or test data.
    data_values : TYPE
        Values of given variable.
    time_values : TYPE
        Time values of given variable.
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
    
    maps_composite = [ [None for _ in range(len(cluster_dates_list[0])) ] for i in range(number_soms)]
    for i in range(len(cluster_dates_list)):
        for j in range(len(cluster_dates_list[i])):
            list_map_temp = []
            for k in range(len(cluster_dates_list[i][j])):
                indx = np.where(time_values[:]==cluster_dates_list[i][j][k])[0][0]
                
                #print(cluster_dates_list[0][0][k])
                #print(time_values[indx])
                #print("")
                list_map_temp += [data_values[indx]] 
                print(time_values[indx])
                print(cluster_dates_list[i][j][k])
                
            list_map_mean = np.mean(list_map_temp, axis=0)
            maps_composite[i][j]=np.reshape(list_map_mean,(nx,ny))
            #print(len(list_map_temp))
    
    return maps_composite

def generate_SOM_composites_Z500_mslp(data_shaded, data_contour, names_som,  som_col, som_row, lon, lat, som_variables_str, test_or_all=None, chosen_map=maps_colormap):
    # Z500 shaded
    # mslp contour
    
    datacrs = ccrs.PlateCarree()
    for i in range(len(data_shaded)):    
        """
        #You will set this to the dimensions of the SOM.  ## DIM OF SOMS
        fig, axs = plt.subplots(nrows=som_row,ncols=som_col, 
                                subplot_kw={'projection':ccrs.LambertConformal(central_longitude=lon.mean(), central_latitude=lat.mean(), standard_parallels=(30, 60))},
                                figsize=(30, 15),facecolor='white') 
        """
        #You will set this to the dimensions of the SOM.  ## DIM OF SOMS
        fig, axs = plt.subplots(nrows=som_row,ncols=som_col, 
                                subplot_kw={'projection': ccrs.PlateCarree()},
                                figsize=(40, 20),facecolor='white') 
        axs=axs.flatten()
        for k in range(len(data_shaded[i])):
            # to check
            # Z500
            lev_start = 5150
            #print(lev_start)             
            lev_stop = 5880
            step = 30 # generalmente è 30
            #print(lev_stop)
            n_cmap = int((lev_stop - lev_start) / step)
            cmap = plt.cm.get_cmap(chosen_map, n_cmap)
            levs = np.arange(lev_start, lev_stop, step)
            norm = mpl.colors.BoundaryNorm(levs, cmap.N)
            
            cs2=axs[k].contourf(lon, lat, data_shaded[i][k],
                              n_cmap, cmap=cmap, norm=norm, levels=levs, extend='both', transform = ccrs.PlateCarree())
            
            # MSLP
            levels = np.arange(950, 1050, 3) # label height
            contour = axs[k].contour(lon, lat, data_contour[i][k], levels, colors='k', transform = ccrs.PlateCarree(), linewidths=1.5)
            plt.clabel(contour, inline=True, fontsize=10, fmt='%1.0f') 
   
            axs[k].set_extent([lon[0], lon[-1], lat[0], lat[-1]], ccrs.PlateCarree())

            axs[k].coastlines()
            axs[k].add_feature(cfeature.BORDERS, linestyle="--") 

            # Title each subplot 
            #axs[k].set_title('Node:'+str(k+1) + " Freq:" + str(frequencies[k,i]), fontsize=18)
            axs[k].set_title('Node:'+str(k+1), fontsize=18)

            plt.tight_layout()
            fig.subplots_adjust(bottom=0.25, top=0.9, left=0.05, right=0.6,
                        wspace=0.05, hspace=0.25)
        
        cbar_ax = fig.add_axes([0.08, 0.2, 0.5, 0.02])
        #cbar=fig.colorbar(cs2,cax=cbar_ax, ticks = np.arange(lev_start, np.abs(lev_start)+lev_step, lev_step*2),orientation='horizontal')
        #cbar=fig.colorbar(cs2,cax=cbar_ax, ticks = levs,orientation='horizontal')
        cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm,
                                     spacing='proportional', ticks=levs, boundaries=levs, format='%1i',  orientation='horizontal')
        
        str_variable = "Z500"
        unit_variable = "m"
        cbar.set_label(str_variable + " (" + unit_variable + ")", fontsize=22)
        
    
        plt.suptitle('Composites of Z500 (shaded) and mSLP (contour). ' + dataset_str + ", SOM of " + som_variables_str + " n" + names_som[i][-2:], x= 0.33 ,fontsize=27)   
        plt.savefig(folderpath + 'Composites_Z500_mSLP_' +names_som[i]+ '_' + test_or_all + '.png', bbox_inches='tight')
        plt.show()

def generate_SOM_composites_Z500_IVT(data_contour, ivte, ivtn, names_som, test_or_all=None, chosen_map="jet"):
    # Z500 contour
    # ivt arrows
    
    datacrs = ccrs.PlateCarree()
    for i in range(len(data_contour)):    
        """
        #You will set this to the dimensions of the SOM.  ## DIM OF SOMS
        fig, axs = plt.subplots(nrows=som_row,ncols=som_col, 
                                subplot_kw={'projection':ccrs.LambertConformal(central_longitude=lon.mean(), central_latitude=lat.mean(), standard_parallels=(30, 60))},
                                figsize=(30, 15),facecolor='white') 
        """
        #You will set this to the dimensions of the SOM.  ## DIM OF SOMS
        fig, axs = plt.subplots(nrows=som_row,ncols=som_col, 
                                subplot_kw={'projection': ccrs.PlateCarree()},
                                figsize=(40, 20),facecolor='white') 
        axs=axs.flatten()
        for k in range(len(data_contour[i])):
            # to check
            # Z500
            lev_start = 5150
            #print(lev_start)             
            lev_stop = 5880
            step = 30 # generalmente è 30
            #print(lev_stop)
            n_cmap = int((lev_stop - lev_start) / step)
            cmap = plt.cm.get_cmap(chosen_map, n_cmap)
            levs = np.arange(lev_start, lev_stop, step)
        
            contour = axs[k].contour(lon, lat, data_contour[i][k], levs, colors='k', transform = ccrs.PlateCarree(), linewidths=1.5)
            plt.clabel(contour, inline=True, fontsize=10, fmt='%1.0f') 
   
            axs[k].set_extent([2, 19, 50, 35], ccrs.PlateCarree())

            axs[k].coastlines()
            axs[k].add_feature(cfeature.BORDERS, linestyle="--") 
            
            #axs[k].streamplot(lon, lat, ivte[i][k], ivtn[i][k], density=0.5, color="white")
            #axs[k].quiver(lon, lat, ivte[i][k], ivtn[i][k])
            
            # Calcola la magnitudine del vettore IVT
            ivt_magnitude = np.sqrt(ivte[i][k]**2 + ivtn[i][k]**2)
            
            #MASK
            """
            # se vuoi utilizzare la maschera
            mask = ivt_magnitude > 100
            ivte_filtered = np.ma.masked_where(~mask, ivte[i][k])
            ivtn_filtered = np.ma.masked_where(~mask, ivtn[i][k])
            ivt_magnitude_filtered = np.ma.masked_where(~mask, ivt_magnitude)
            quiver = axs[k].quiver(lon, lat, ivte_filtered, ivtn_filtered, ivt_magnitude_filtered, 
                                   cmap=cmap_custom, norm=norm_custom, transform=ccrs.PlateCarree(), angles='xy', scale_units='xy', scale=1.,headwidth=1, density=0.5)
            
            """
            # Skip
            n_skip= 4
            skip = (slice(None, None, n_skip), slice(None, None, n_skip))
            
            # IVT Colormaps
            lev_start = 0          
            lev_stop = 501
            step = 25
            n_cmap = int((lev_stop - lev_start) / step)
            cmap = plt.cm.get_cmap(chosen_map, n_cmap)
            bounds = np.arange(lev_start, lev_stop, step)
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            
            # Disegna il quiver plot con i colori personalizzati
            quiver = axs[k].quiver(lon[::n_skip], lat[::n_skip], ivte[i][k][skip], ivtn[i][k][skip], ivt_magnitude[skip], cmap=cmap, norm=norm, transform=ccrs.PlateCarree(),
                       headwidth=3, scale=250, headlength=5, width=0.08, units="xy", edgecolor="k", linewidths=0.6)
            #quiver = axs[k].quiver(lon, lat, ivte_filtered, ivtn_filtered, ivt_magnitude_filtered, 
                                   #cmap=cmap_custom, norm=norm_custom, transform=ccrs.PlateCarree(), angles='xy', scale_units='xy', scale=1.,headwidth=1, density=0.5)

            # Title each subplot 
            #axs[k].set_title('Node:'+str(k+1) + " Freq:" + str(frequencies[k,i]), fontsize=18)
            axs[k].set_title('Node:'+str(k+1), fontsize=18)

            plt.tight_layout()
            fig.subplots_adjust(bottom=0.25, top=0.9, left=0.05, right=0.6,
                        wspace=0.05, hspace=0.25)
        
        # Colorbar per il quiver plot
        cbar_ax = fig.add_axes([0.08, 0.2, 0.5, 0.02])
        cbar2 = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm,
                                          spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i', orientation='horizontal')
        cbar2.set_label(r'IVT [$kg \cdot m^{-1} \cdot s^{-1}$)]', fontsize=22)
        
        
        plt.suptitle('Composites of Z500 (contour) and IVT (arrows). ' + dataset_str + ", SOM of " + som_variables_str + " n" + names_som[i][-2:], x= 0.33 ,fontsize=27)   
        #plt.savefig(folderpath + 'Composites_Z500_IVT_' +names_som[i]+ '_' + test_or_all + '.png', bbox_inches='tight')
        plt.show()

def SOM_maps_composites_with_ref_nearest(data_values_som, time_values_som, data_values, time_values, som_shape, number_soms, filepath_som, som_names, som_col, som_row):
    """
    Generate composites of data variable for the differents nodes of the differents soms

    Parameters
    ----------
    data_values_som : .npy
        Values of train data or test data.
    time_values_som : .npy
        Time values of train data or test data.
    data_values : TYPE
        Values of given variable.
    time_values : TYPE
        Time values of given variable.
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
            
    maps_composite = [ [None for _ in range(len(cluster_dates_list[0])) ] for i in range(number_soms)]
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
                
            list_map_mean = np.mean(list_map_temp, axis=0)
            print("shape")
            print(list_map_mean.shape)
            #maps_composite[i][j] = list_map_mean[0,:,:]
           
            try:
                maps_composite[i][j]=np.reshape(list_map_mean,(nx,ny))
            except ValueError:
                maps_composite[i][j]=np.zeros((nx,ny))
            #print(len(list_map_temp))
    
    return maps_composite

def generate_SOM_composites_pr(data, names_som, som_col, som_row, dataset_name = None, test_or_all=None):
    
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
            levs = np.arange(0, 60, 2)
            n_cmap = len(levs)
            norm = mpl.colors.BoundaryNorm(levs, n_cmap)
            cmap = plt.cm.get_cmap(precip_colormap25, n_cmap)
            cs2=axs[k].contourf(lon, lat, data[i][k],
                              n_cmap, cmap=cmap, norm=norm, transform = ccrs.PlateCarree(), extend='both')               
                

            #axs[k].set_extent([4, 19, 50, 35], ccrs.PlateCarree()) # all Italy
            #domain = "IT"
            axs[k].set_extent([6.5, 14.2, 47.5, 41.9], ccrs.PlateCarree()) # ARCIS domain
            domain = "NI"


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
        print("nella funzione ", som_variables_str)
        plt.suptitle('Composites of pr (' + dataset_name + " - " + test_or_all + " data) "+ "SOM (" + som_variables_str + ") n." + names_som[i][-2:] , x= 0.33 ,fontsize=28)   
        plt.savefig(folderpath + 'Composites_pr_' + dataset_name + "_" +names_som[i]+ '_' + test_or_all + '_' + domain +'_v2.png', bbox_inches='tight')
        plt.show()

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