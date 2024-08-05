# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 17:31:55 2024

@author: Criss
"""
import numpy as np
import pandas as pd
#import glob
import xarray as xr
#from xarray import DataArray

### Figures : maps and altitude maps
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap,LinearSegmentedColormap

# pip install mpl-scatter-density
#import mpl_scatter_density # adds projection='scatter_density'

# Map figure
import warnings; warnings.filterwarnings("ignore")
import cartopy.crs as ccrs

crs = ccrs.PlateCarree()

from mpl_toolkits.axes_grid1 import make_axes_locatable

#import cartopy.crs as ccrs
import cartopy.feature as cfeature
#import proplot
from matplotlib import ticker


# Statistics
from scipy import stats

from scipy.constants import g

#from SOM_variable_file import *
#print(som_variables_str)


# COLORMAP
new_colormap = [
    "#ecf5ff",     # celeste più chiaro
   # "#d9ecff",    #new
    "#a1cff7",    #new
    "#5ca8e5",    #new
    "#2476b5",    #new
    "#0c4b78",    #new
    "#00334c",    #new
    "#005259",    #new
    "#008c69",    #new
    "#00cc44",
    "#95ff00",
    "#ffff00",
    "#ffd400",
    "#ffaa00",
    "#ff7f00",
    "#ff5500",
    "#ff2a00",
    "#f20c1f",
    "#cc1461",
    "#eb1cb7",    #new
    "#be21cc",
    "#8613bf",
    "#5f19a6",
    "#330067",    #new
]

precip_colormap = mpl.colors.ListedColormap(new_colormap)
# Riduzione della saturazione
lighter_colormap = mpl.colors.ListedColormap(
    [mpl.colors.to_rgba(color, alpha=0.75) for color in new_colormap]
)

colormap25_pr = [
    "#ecf5ff",     # celeste più chiaro
   # "#d9ecff",
   "#d9ecff",
    "#a1cff7",
    "#5ca8e5",
    "#2476b5",
    "#0c4b78",
    "#00334d",    #modified
    "#003d52",    #new
    "#005259",
    "#008c69",
    "#00cc44",
    "#95ff00",
    "#ffff00",
    "#ffd400", 
    "#ffaa00",
    "#ff7f00",
    "#ff5500",
    "#ff2a00",
    "#f20c1f",
    "#cc1461",
    "#eb1cb7",
    "#be21cc",
    "#8613bf",
    "#6419a6",    #modified
    "#450a80",    #new
    "#330066",
    ]

precip_colormap25 = mpl.colors.ListedColormap(colormap25_pr)
# Riduzione della saturazione
lighter_colormap25 = mpl.colors.ListedColormap(
    [mpl.colors.to_rgba(color, alpha=0.75) for color in colormap25_pr]
)

colormap25 = [
    #"#ecf5ff",     # celeste più chiaro
    #"#d9ecff", # troppo chiaro per le mappe
    "#a1cff7",
    "#5ca8e5",
    "#2476b5",
    "#0c4b78",
    "#00334d",    #modified
    "#003d52",    #new
    "#005259",
    "#008c69",
    "#00cc44",
    "#95ff00",
    "#ffff00",
    "#ffd400", 
    "#ffaa00",
    "#ff7f00",
    "#ff5500",
    "#ff2a00",
    "#f20c1f",
    "#cc1461",
    "#eb1cb7",
    "#be21cc",
    "#8613bf",
    "#6419a6",    #modified
    "#450a80",    #new
    "#330066",
    ]

maps_colormap = mpl.colors.ListedColormap(colormap25)
# Riduzione della saturazione
maps_colormap_lighter = mpl.colors.ListedColormap(
    [mpl.colors.to_rgba(color, alpha=0.75) for color in colormap25]
)

# OPEN DATA
def ERA5_IVT(path_file, domain=None, sel_time=None):
    """
    To load IVT data
    Parameters
    ----------
    path_file : string, path for the file
    domain : string, optional
        EU2 or IT . The default is None.
    sel_time : string, optional
        "time1" for 1991-2022 or "time2" for 1985-2019 The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
  
    ds = xr.open_mfdataset(path_file)
    variable_list = list(ds.keys())
    print(variable_list)
    # preprocessing 

    if domain == "EU3":
        dset = ds.sel(lat = slice(60,25), lon = slice(-13,35))
    elif domain=="IT":
        dset = ds.sel(lat = slice(48,36), lon = slice(6,19)) # italy
    else :
        dset = ds
        print("All domain")
    
    if sel_time == "time1":
        dset = dset.sel(time=slice("1991","2022"))
        
    elif sel_time == "time2":
        dset = dset.sel(time=slice("1985","2019"))
        
    try :
        print("is the first variable time bounds and the second the main data? if not check")
        Time_Bounds = dset[variable_list[0]].values
        Main_data = dset[variable_list[1]].values
    except :
        print("check the dataset's variable")
        return 0
        
    Time_data = dset["time"].values
    Xlat = dset["lat"].values
    Xlong = dset["lon"].values
    
    return Main_data, Time_data, Xlat, Xlong, Time_Bounds

def simple_open_data(path_file):
    dset = xr.open_mfdataset(path_file)
    variable_list = list(dset.keys())
    print("The variables are: ", variable_list)
    #print(dset)
    try :
        Time_data = dset[variable_list[0]].values
        Main_data = dset[variable_list[1]].values
    except :
        print("check the dataset's variable")
        return 0
        
    Xlat = dset["lat"].values
    Xlong = dset["lon"].values
    
    return Main_data, Time_data, Xlat, Xlong

def simple_open_data_ARCIS(path_file):
    # ok for  MSWEP
    dset = xr.open_mfdataset(path_file)
    variable_list = list(dset.keys())
    print("The variables are: ", variable_list)
    #print(dset)
    try :
        Main_data = dset[variable_list[0]].values
    except :
        print("check the dataset's variable")
        return 0
    
    Time_data = dset["time"].values
    Xlat = dset["lat"].values
    Xlong = dset["lon"].values
    
    return Main_data, Time_data, Xlat, Xlong

def open_data_fld_mean(path_file):
    dset = xr.open_mfdataset(path_file)
    variable_list = list(dset.keys())
    print("The variables are: ", variable_list)
    #print(dset)
    try :
        Main_data = dset[variable_list[0]].values
    except :
        print("check the dataset's variable")
        return 0
    Time_data = dset["time"].values
    
    return Main_data, Time_data

"""
def open_data_and_select_lat_lon(path_file):

    ds = xr.open_mfdataset(path_file, chunks={'Times': '900MB'})
    #dset = ds.sel(lat = slice(60,36), lon = slice(-10,19)) # anche spagna e fino a scozia
    dset = ds.sel(lat = slice(48,36), lon = slice(6,19)) # italy
    
    variable_list = list(dset.keys())
    print("The variables are: ", variable_list)
    #print(dset)
    try :
        Main_data = dset[variable_list[0]].values
    except :
        print("check the dataset's variable")
        return 0
    
    Time_data = dset["time"].values
    Xlat = dset["lat"].values
    Xlong = dset["lon"].values
    
    return Main_data, Time_data, Xlat, Xlong
"""
    #dset = ds.sel(lat = slice(60,35), lon = slice(-15,30)) # più grande
    #dset = ds.sel(lat = slice(60,36), lon = slice(-10,19)) # anche spagna e fino a scozia
    #dset = ds.sel(lat = slice(48,36), lon = slice(6,19)) # italy
def open_data_and_select_lat_lon(path_file, lat_slice=None, lon_slice=None):

    ds = xr.open_mfdataset(path_file, chunks={'Times': '900MB'})
    
    if lat_slice is not None and lon_slice is not None:
        dset = ds.sel(lat=lat_slice, lon=lon_slice)
    else:
        dset = ds
    
    variable_list = list(dset.keys())
    print("The variables are: ", variable_list)
    
    try:
        Main_data = dset[variable_list[0]].values
    except:
        print("check the dataset's variable")
        return 0
    
    Time_data = dset["time"].values
    Xlat = dset["lat"].values
    Xlong = dset["lon"].values
    
    return Main_data, Time_data, Xlat, Xlong

#def select_latlon(ds):
#    #return ds.sel(lat = slice(60,36), lon = slice(-10,19)) #EU_1
#    return ds.sel(lat = slice(60,35), lon = slice(-15,30))#EU_2

def select_latlon(ds):
    #return ds.sel(lat = slice(60,36), lon = slice(-10,19)) #EU_1
    #return ds.sel(lat = slice(60,35), lon = slice(-15,30))#EU_2
    return ds.sel(lat = slice(60,25), lon = slice(-13,35))#EU_3 (aggiornato)
    #return ds.sel(lat = slice(48,36), lon = slice(5,19))# Italy


# DATAFRAME MANIPULATIONS
def dataframe_cut(path_csv_file, first_year, last_year):
    # function to cut a selected dataset from a bigger one
    # check 
    # - first column is the column of times
    # - the format of the dates
    
    data_raw= pd.read_csv(path_csv_file)
    label_first_column = data_raw.columns[0] #get the label of the first column

    data = data_raw.rename(columns={ label_first_column : "Time"}) #change the label
    
    data['Time'] = pd.to_datetime(data['Time'])
    #data['Time'] = data['Time'].apply(pd.to_datetime)
    # do the time mask
    time_mask = (data['Time'].dt.year >= first_year) & \
                (data['Time'].dt.year <= last_year)
    # select the data
    data[time_mask]
    choseInd = [ind for ind in data[time_mask].index]
    
    df_select = data.loc[choseInd]
    
    return data_raw, df_select

def dataframe_cut_xlsx(path_xlsx_file, first_year, last_year, exclude_last_row=False):
    # function to cut a selected dataset from a bigger one
    # check 
    # - first column is the column of times
    # - the format of the dates
    data_raw= pd.read_excel(path_xlsx_file)
    data_tmp = data_raw
    
    if exclude_last_row == True:
        data_tmp = data_raw.iloc[:-1]
        
    label_first_column = data_tmp.columns[0] #get the label of the first column

    data = data_tmp.rename(columns={ label_first_column : "Time"}) #change the label
    
    data['Time'] = pd.to_datetime(data['Time'])
    #data['Time'] = data['Time'].apply(pd.to_datetime)
    # do the time mask
    time_mask = (data['Time'].dt.year >= first_year) & \
                (data['Time'].dt.year <= last_year)
    # select the data
    data[time_mask]
    choseInd = [ind for ind in data[time_mask].index]
    
    df_select = data.loc[choseInd]
    
    return data_raw, df_select


def dataframe_cut_xlsx_sheet(path_xlsx_file, sheet_name, first_year, last_year, exclude_last_row=False):
    # function to cut a selected dataset from a bigger one
    # check 
    # - first column is the column of times
    # - the format of the dates
    data_raw= pd.read_excel(path_xlsx_file, sheet_name=sheet_name)
    data_tmp = data_raw
    
    if exclude_last_row == True:
        data_tmp = data_raw.iloc[:-1]
        
    label_first_column = data_tmp.columns[0] #get the label of the first column

    data = data_tmp.rename(columns={ label_first_column : "Time"}) #change the label
    
    data['Time'] = pd.to_datetime(data['Time'])
    #data['Time'] = data['Time'].apply(pd.to_datetime)
    # do the time mask
    time_mask = (data['Time'].dt.year >= first_year) & \
                (data['Time'].dt.year <= last_year)
    # select the data
    data[time_mask]
    choseInd = [ind for ind in data[time_mask].index]
    
    df_select = data.loc[choseInd]
    
    return data_raw, df_select



def dataframe_extremes_xlsx(pd_dataframe, label_col_extreme):
    df = pd_dataframe
    #label_first_column = df.columns[-2]
    df_select = df.loc[df[label_col_extreme] == True]
    return df_select

def dataframe_to_datatime(path_csv_file):
    # Function to set the column of dates as "Time"
    # it also converts into datatime
    # check 
    # - first column is the column of times
    # - the format of the dates
    
    data_raw= pd.read_csv(path_csv_file)
    label_first_column = data_raw.columns[0] #get the label of the first column
    data = data_raw.rename(columns={ label_first_column : "Time"}) #change the label
    data['Time'] = pd.to_datetime(data['Time'])
    
    return data              

def add_pr_max_column(dataset, idx_first_c, idx_last_c):
    idx_stop = idx_last_c + 1
    df_cutted = dataset.iloc[:, idx_first_c: idx_stop]
    pr_max_c = df_cutted.max(axis=1)
    dataset["PrMax"]=pr_max_c
    #new_df = dataset.insert[1, "Pr Max", pr_max_c]
    
    return dataset

def dataframe_pr_cat(df_pr, df_dates, N_CAT):
    extreme_dates = df_dates["Time"]
    extreme_dates_CAT = df_dates[df_dates["Cat"].isin([N_CAT])]["Time"]
    extreme_list = extreme_dates.tolist()
    extreme_list_CAT = extreme_dates_CAT.tolist()
    
    df_pr_all_CAT=df_pr[df_pr["Time"].isin(extreme_list)]
    df_pr_selected_CAT=df_pr[df_pr["Time"].isin(extreme_list_CAT)]
    
    return df_pr_all_CAT, df_pr_selected_CAT

def save_dataframe_csv(dataframe, output_path, file_name, sorted_by, asc = False):
    # asc indica l'ordine in cui ordina il file prima di salvarlo
    # se asc è True lo mette in ordine crescente, se è False è discendente
    dataframe.to_csv(output_path + "/" + file_name + ".csv", index=False)
    dataframe_sorted = dataframe.sort_values(sorted_by, ascending= asc)
    dataframe_sorted.to_csv(output_path + "/" + file_name + "_sorted_by_ " + str(sorted_by) + ".csv", index=False)
    return dataframe_sorted



# MAPS

def map_single(fig, data_array, Xlat, Xlong, title_fig, bar_unit=r"angular coefficient m $[year^{-1}]$", vmin=None, vmax=None, chosen_map="Reds"):
    # create a figure outside before such as 
    # fig=plt.figure(figsize=(14,7))
    
    ## extremes of colormap
    data_not_nan = np.nan_to_num(data_array)
    if vmax is None :
        max_value = data_not_nan.max()  
    else :
        max_value = vmax
        
    if vmin is None :
        min_value = data_not_nan.min()  
    else :
        min_value = vmin
        
    #print(m_array.shape)
    #print(xlat.shape)
    #print(xlong.shape)

    #Figure
    ax = fig.add_subplot(projection=crs) #projections
    ax.add_feature(cfeature.COASTLINE)
    plt.title(title_fig, fontsize=18, fontweight='bold')
    #im = plt.pcolormesh(Xlong, Xlat, m_not_nan, cmap=chosen_map)
    im = plt.pcolormesh(Xlong, Xlat, data_not_nan, cmap=chosen_map, vmin=min_value, vmax=max_value)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False, linewidth=0.33, color='k',alpha=0.5)
    gl.right_labels = gl.top_labels = False
    #gl.ylocator = ticker.FixedLocator([44,46,48])
    #gl.xlocator = ticker.FixedLocator([-100, -95, -90, -85, -80, -75, -70])
    
    # Colorbar
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    plt.colorbar(im, cax=cax, label=bar_unit) # Similar to fig.colorbar(im, cax = cax)
    
    # Altitutude
    #levels = np.arange(0, 3000, 500)
    #contour = ax.contour(Xlong, Xlat, hgt_2d_no_sea, levels, colors='k', linewidths=1)
    #plt.clabel(contour, inline=True, fontsize=10, fmt='%1.0f') 
    
    plt.show()

def extreme_10_maps(data_maps, extreme_dates, Xlong, Xlat, fig_title, chosen_map="Reds", bar_unit="atm"):
    #
    
    # è una aggiunta
    data_numpy = np.array(data_maps)
    data_not_nan = data_numpy[~np.isnan(data_numpy)]
    
    #min_value = np.min(data_maps)
    min_value = np.min(data_not_nan)
    print("min value is ", min_value)
    #max_value = np.max(data_maps)
    max_value = np.max(data_not_nan)
    print("max value is ", max_value)
    
    extreme_dates_list = extreme_dates.tolist()
    # Figure - Layer
    #ar = 1.5  # initial aspect ratio for first trial
    #wi = 10    # width in inches
    #hi = wi * ar  # height in inches

    gs = gridspec.GridSpec(5,2)
    cmap = plt.cm.get_cmap(chosen_map,12)
    
    # Set figsize using wi and hi
    fig = plt.figure(figsize=(10, 15))    
    fig.suptitle(fig_title, fontsize=20)
    
    # Figure - base
    ax = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
    ax.set_title("1. " + str(extreme_dates_list[0]), fontsize=12)
    im_0 = plt.contourf(Xlong, Xlat, data_maps[0], vmin=min_value, vmax=max_value, cmap=cmap)
    ax.add_feature(cfeature.COASTLINE)
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False, linewidth=0.33, color='k', alpha=0.5)
    gl.right_labels = gl.top_labels = False
    #gl.ylocator = ticker.FixedLocator([44, 46, 48])
    #gl.xlocator = ticker.FixedLocator([4, 9, 14, 19])

    # Figure - height
    #levels = np.arange(0, 3000, 500) # label height
    #contour = ax.contour(xlong, xlat, hgt_2d_no_sea, levels, colors='k', linewidths=1)
    
    for k in range(1, 10):
        # Figure - base
        ax = fig.add_subplot(gs[k], projection=ccrs.PlateCarree())
        ax.set_title(str(k+1) + ". " + str(extreme_dates_list[k]), fontsize=12)
        im = plt.contourf(Xlong, Xlat, data_maps[k], vmin=min_value, vmax=max_value, cmap=cmap)
        ax.add_feature(cfeature.COASTLINE)
         # axes
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False, linewidth=0.33, color='k', alpha=0.5)
        gl.right_labels = gl.top_labels = False
        #gl.ylocator = ticker.FixedLocator([44, 46, 48])
        #gl.xlocator = ticker.FixedLocator([4, 9, 14, 19])
        
        # Figure - height
        #levels = np.arange(0, 3000, 500) # label height
        #contour = ax.contour(xlong, xlat, hgt_2d_no_sea, levels, colors='k', linewidths=1)
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([1, 0.05, 0.02, 0.8])
    
    cbar = fig.colorbar(im_0, cax=cbar_ax, label= bar_unit)

    # Get proper ratio here
    #xmin, xmax = ax.get_xbound()
    #ymin, ymax = ax.get_ybound()
    #y2x_ratio = (ymax-ymin)/(xmax-xmin)

    #print("y2x_ratio: "+ str(y2x_ratio)) 

    #fig.set_figheight(wi * y2x_ratio + 1.)

    gs.tight_layout(fig, pad=1.2)
    fig.savefig("Grafici/" + str(fig_title) + ".png", bbox_inches="tight")
    plt.show()
"""
def extreme_1_maps_contour(data_maps, cont_maps, fig_title, Xlong, Xlat, chosen_map="Reds", bar_unit="Z [m]", vmin=None, vmax=None):

    ## extremes of colormap
    #data_not_nan = np.nan_to_num(data_maps)
    #if vmax is None :
    #    max_value = data_not_nan.max()  
    #else :
    #    max_value = vmax
        
    #if vmin is None :
    #    min_value = data_not_nan.min()  
    #else :
    #    min_value = vmin
    
    min_value = np.min(data_maps)
    print("min value is ", min_value)
    max_value = np.max(data_maps)
    print("max value is ", max_value)
    
    print("cont_min", np.min(cont_maps))
    print("cont_max", np.max(cont_maps))
        
    # settings colorbar
    cmap = plt.cm.get_cmap(chosen_map, 19)
    bounds = np.arange(4890, 5980, 60)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    #Figure
    fig=plt.figure(figsize=(14,7))
    ax = fig.add_subplot(projection=crs) #projections
    ax.add_feature(cfeature.COASTLINE)
    plt.title(fig_title, fontsize=18, fontweight='bold')
    #im = plt.pcolormesh(Xlong, Xlat, m_not_nan, cmap=chosen_map)
    im = plt.contourf(Xlong, Xlat, data_maps, 19, cmap=cmap, norm=norm)
    #im = plt.pcolormesh(Xlong, Xlat, data_not_nan, cmap=chosen_map, vmin=min_value, vmax=max_value)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False, linewidth=0.33, color='k',alpha=0.5)
    gl.right_labels = gl.top_labels = False
    #gl.ylocator = ticker.FixedLocator([44,46,48])
    #gl.xlocator = ticker.FixedLocator([-100, -95, -90, -85, -80, -75, -70])

    # Figure - contour
    levels = np.arange(950, 1050, 3) # label height
    contour = ax.contour(Xlong, Xlat, cont_maps, levels, colors='k', linewidths=1)
    plt.clabel(contour, inline=True, fontsize=10, fmt='%1.0f') 
    
    # Colorbar
    #cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    #plt.colorbar(im, cax=cax, label=bar_unit) # Similar to fig.colorbar(im, cax = cax)
    
        
    fig.subplots_adjust(right=0.8)
    cbar_width = 0.02  # Larghezza dell'asse della color bar
    cbar_height = 0.8  # Altezza dell'asse della color bar
    # Aggiungi l'asse della color bar
    cbar_ax = fig.add_axes([1.005, 0.1, cbar_width, cbar_height])  # Aggiustamento della posizione e delle dimensioni

    # Aggiungi la color bar
    cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm,
                                 spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i',  orientation='vertical', label=bar_unit)


    fig.tight_layout()
    
    plt.show()
    
def extreme_1_maps_contour(data_maps, cont_maps, fig_title, Xlong, Xlat, chosen_map="Reds", bar_unit="Z [m]", vmin=None, vmax=None):

    ## extremes of colormap
    #data_not_nan = np.nan_to_num(data_maps)
    #if vmax is None :
    #    max_value = data_not_nan.max()  
    #else :
    #    max_value = vmax
        
    #if vmin is None :
    #    min_value = data_not_nan.min()  
    #else :
    #    min_value = vmin
    
    min_value = np.min(data_maps)
    print("min value is ", min_value)
    max_value = np.max(data_maps)
    print("max value is ", max_value)
    
    min_value_c = np.min(cont_maps)
    max_value_c = np.max(cont_maps)
    print("cont_min", np.min(cont_maps))
    print("cont_max", np.max(cont_maps))
        
    n_cmap= int((max_value - min_value) /30)
    print(n_cmap)
    # settings colorbar
    cmap = plt.cm.get_cmap(chosen_map, n_cmap)
    bounds = np.arange(min_value, max_value, 30)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    #Figure
    fig=plt.figure(figsize=(14,7))
    ax = fig.add_subplot(projection=crs) #projections
    ax.add_feature(cfeature.COASTLINE)
    plt.title(fig_title, fontsize=18, fontweight='bold')
    #im = plt.pcolormesh(Xlong, Xlat, m_not_nan, cmap=chosen_map)
    im = plt.contourf(Xlong, Xlat, data_maps, n_cmap, cmap=cmap, norm=norm)
    #im = plt.pcolormesh(Xlong, Xlat, data_not_nan, cmap=chosen_map, vmin=min_value, vmax=max_value)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False, linewidth=0.33, color='k',alpha=0.5)
    gl.right_labels = gl.top_labels = False
    #gl.ylocator = ticker.FixedLocator([44,46,48])
    #gl.xlocator = ticker.FixedLocator([-100, -95, -90, -85, -80, -75, -70])

    # Figure - contour
    levels = np.arange(min_value_c, max_value_c, 3) # label height
    contour = ax.contour(Xlong, Xlat, cont_maps, levels, colors='k', linewidths=1)
    plt.clabel(contour, inline=True, fontsize=10, fmt='%1.0f') 
    
    # Colorbar
    #cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    #plt.colorbar(im, cax=cax, label=bar_unit) # Similar to fig.colorbar(im, cax = cax)
    
        
    fig.subplots_adjust(right=0.8)
    cbar_width = 0.02  # Larghezza dell'asse della color bar
    cbar_height = 0.8  # Altezza dell'asse della color bar
    # Aggiungi l'asse della color bar
    cbar_ax = fig.add_axes([1.005, 0.1, cbar_width, cbar_height])  # Aggiustamento della posizione e delle dimensioni

    # Aggiungi la color bar
    cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm,
                                 spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i',  orientation='vertical', label=bar_unit)


    fig.tight_layout()
    fig.savefig("Grafici/" + str(fig_title) + ".png", bbox_inches="tight")
    plt.show()
    
    return 1
"""
def extreme_1_maps_contour(data_maps, cont_maps, fig_title, Xlong, Xlat, chosen_map=lighter_colormap, bar_unit="Z [m]", vmin=None, vmax=None):

    ## extremes of colormap
    #data_not_nan = np.nan_to_num(data_maps)
    #if vmax is None :
    #    max_value = data_not_nan.max()  
    #else :
    #    max_value = vmax
        
    #if vmin is None :
    #    min_value = data_not_nan.min()  
    #else :
    #    min_value = vmin
    
    min_value = np.min(data_maps)
    print("min value is ", min_value)
    max_value = np.max(data_maps)
    print("max value is ", max_value)
    
    min_value_c = np.min(cont_maps)
    max_value_c = np.max(cont_maps)
    print("cont_min", np.min(cont_maps))
    print("cont_max", np.max(cont_maps))
        
    n_cmap = int((5900 - 5180) /30)
    #print(n_cmap)
    # settings colorbar
    cmap = plt.cm.get_cmap(chosen_map, n_cmap)
    bounds = np.arange(5180, 5900, 30)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    #Figure
    fig=plt.figure(figsize=(14,7))
    ax = fig.add_subplot(projection=crs) #projections
    ax.add_feature(cfeature.COASTLINE)
    plt.title(fig_title, fontsize=18, fontweight='bold')
    #im = plt.pcolormesh(Xlong, Xlat, m_not_nan, cmap=chosen_map)
    im = plt.contourf(Xlong, Xlat, data_maps, n_cmap, cmap=cmap, norm=norm)
    #im = plt.pcolormesh(Xlong, Xlat, data_not_nan, cmap=chosen_map, vmin=min_value, vmax=max_value)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False, linewidth=0.33, color='k',alpha=0.5)
    gl.right_labels = gl.top_labels = False
    #gl.ylocator = ticker.FixedLocator([44,46,48])
    #gl.xlocator = ticker.FixedLocator([-100, -95, -90, -85, -80, -75, -70])

    # Figure - contour
    levels = np.arange(min_value_c, max_value_c, 3) # label height
    contour = ax.contour(Xlong, Xlat, cont_maps, levels, colors='k', linewidths=1)
    plt.clabel(contour, inline=True, fontsize=10, fmt='%1.0f') 
    
    # Colorbar
    #cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    #plt.colorbar(im, cax=cax, label=bar_unit) # Similar to fig.colorbar(im, cax = cax)
    
        
    fig.subplots_adjust(right=0.8)
    cbar_width = 0.02  # Larghezza dell'asse della color bar
    cbar_height = 0.8  # Altezza dell'asse della color bar
    # Aggiungi l'asse della color bar
    cbar_ax = fig.add_axes([1.005, 0.1, cbar_width, cbar_height])  # Aggiustamento della posizione e delle dimensioni

    # Aggiungi la color bar
    cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm,
                                 spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i',  orientation='vertical', label=bar_unit)


    fig.tight_layout()
    fig.savefig("Grafici/" + str(fig_title) + ".png", bbox_inches="tight")
    plt.show()
    
    return 1        
        
def extreme_1_maps_contour_std(data_maps, cont_maps, fig_title, Xlong, Xlat, chosen_map=lighter_colormap, bar_unit="Z [m]", vmin=None, vmax=None, rel=False):

    ## extremes of colormap
    #data_not_nan = np.nan_to_num(data_maps)
    #if vmax is None :
    #    max_value = data_not_nan.max()  
    #else :
    #    max_value = vmax
        
    #if vmin is None :
    #    min_value = data_not_nan.min()  
    #else :
    #    min_value = vmin
    
    min_value = np.min(data_maps)
    print("min value is ", min_value)
    max_value = np.max(data_maps)
    print("max value is ", max_value)
    
    min_value_c = np.min(cont_maps)
    max_value_c = np.max(cont_maps)
    print("cont_min", np.min(cont_maps))
    print("cont_max", np.max(cont_maps))
    
    if rel==True:
        min_value=0
        max_value=5.25
        n_cmap= int((max_value - min_value) /0.25)
        #print(n_cmap)
        cmap = plt.cm.get_cmap(chosen_map, n_cmap)
        bounds = np.arange(min_value, max_value, 0.25)
    else :
        min_value = 10.
        max_value = 210.
        n_cmap= int((max_value - min_value) /10)
        #print(n_cmap)
        # settings colorbar
        cmap = plt.cm.get_cmap(chosen_map, n_cmap)
        bounds = np.arange(min_value, max_value, 10.)

    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    #Figure
    fig=plt.figure(figsize=(14,7))
    ax = fig.add_subplot(projection=crs) #projections
    ax.add_feature(cfeature.COASTLINE)
    plt.title(fig_title, fontsize=18, fontweight='bold')
    #im = plt.pcolormesh(Xlong, Xlat, m_not_nan, cmap=chosen_map)
    im = plt.contourf(Xlong, Xlat, data_maps, n_cmap, cmap=cmap, norm=norm)
    #im = plt.pcolormesh(Xlong, Xlat, data_not_nan, cmap=chosen_map, vmin=min_value, vmax=max_value)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False, linewidth=0.33, color='k',alpha=0.5)
    gl.right_labels = gl.top_labels = False
    #gl.ylocator = ticker.FixedLocator([44,46,48])
    #gl.xlocator = ticker.FixedLocator([-100, -95, -90, -85, -80, -75, -70])

    # Figure - contour
    if rel==True:
        levels = np.arange(min_value_c, max_value_c, 0.1) # label height
    else :
        levels = np.arange(min_value_c, max_value_c, 1) # label height
    
    contour = ax.contour(Xlong, Xlat, cont_maps, levels, colors='k', linewidths=1)
    plt.clabel(contour, inline=True, fontsize=10, fmt='%.2f') 
    
    # Colorbar
    #cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    #plt.colorbar(im, cax=cax, label=bar_unit) # Similar to fig.colorbar(im, cax = cax)
    
        
    fig.subplots_adjust(right=0.8)
    cbar_width = 0.02  # Larghezza dell'asse della color bar
    cbar_height = 0.8  # Altezza dell'asse della color bar
    # Aggiungi l'asse della color bar
    cbar_ax = fig.add_axes([1.005, 0.1, cbar_width, cbar_height])  # Aggiustamento della posizione e delle dimensioni

    # Aggiungi la color bar
    cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm,
                                 spacing='proportional', ticks=bounds, boundaries=bounds, format='%.2f',  orientation='vertical', label=bar_unit)


    fig.tight_layout()
    fig.savefig("Grafici/" + str(fig_title) + ".png", bbox_inches="tight")
    plt.show()
    
    return 1

def extreme_10_maps_contour(data_maps, cont_maps, extreme_dates, Xlong, Xlat, fig_title, chosen_map=lighter_colormap, bar_unit="atm"):
    #
    min_value = np.min(data_maps)
    print("min value is ", min_value)
    max_value = np.max(data_maps)
    print("max value is ", max_value)
    
    print("cont_min", np.min(cont_maps))
    print("cont_max", np.max(cont_maps))
    
    extreme_dates_list = extreme_dates.tolist()
    # Figure - Layer
    #ar = 1.5  # initial aspect ratio for first trial
    #wi = 10    # width in inches
    #hi = wi * ar  # height in inches

    gs = gridspec.GridSpec(5,2)
    cmap = plt.cm.get_cmap(chosen_map, 19)
    #cmap = chosen_map
    bounds = np.arange(4890, 5980, 60)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    #new_norm = mpl.colors.BoundaryNorm(new_bounds,new_cmap.N)
    
    # Set figsize using wi and hi
    fig = plt.figure(figsize=(15, 22))    
    fig.suptitle(fig_title, fontsize=20)
    
    # Figure - base
    ax = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
    ax.set_title("1. " + str(extreme_dates_list[0]), fontsize=12)
    im_0 = plt.contourf(Xlong, Xlat, data_maps[0], 19, cmap=cmap, norm=norm)
    ax.add_feature(cfeature.COASTLINE)
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False, linewidth=0.33, color='k', alpha=0.5)
    gl.right_labels = gl.top_labels = False
    #gl.ylocator = ticker.FixedLocator([44, 46, 48])
    #gl.xlocator = ticker.FixedLocator([4, 9, 14, 19])

    #Figure - contour
    levels = np.arange(950, 1050, 3) # label height
    contour = ax.contour(Xlong, Xlat, cont_maps[0], levels, colors='k', linewidths=1)
    plt.clabel(contour, inline=True, fontsize=10, fmt='%1.0f')
    
    for k in range(1, 10):
        # Figure - base
        ax = fig.add_subplot(gs[k], projection=ccrs.PlateCarree())
        ax.set_title(str(k+1) + ". " + str(extreme_dates_list[k]), fontsize=12)
        im = plt.contourf(Xlong, Xlat, data_maps[k], cmap=cmap, norm=norm)
        ax.add_feature(cfeature.COASTLINE)
         # axes
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False, linewidth=0.33, color='k', alpha=0.5)
        gl.right_labels = gl.top_labels = False
        #gl.ylocator = ticker.FixedLocator([44, 46, 48])
        #gl.xlocator = ticker.FixedLocator([4, 9, 14, 19])
        
        # Figure - height
        levels = np.arange(948, 1046, 3) # label height
        contour = ax.contour(Xlong, Xlat, cont_maps[k], levels, colors='k', linewidths=1)
        plt.clabel(contour, inline=True, fontsize=10, fmt='%1.0f')
        
        
    fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([1, 0.05, 0.02, 0.8])
    #cax, kw = mpl.colorbar.make_axes(cbar_ax, orientation='vertical')
    #cbar = fig.colorbar(im_0, cax=cbar_ax, label= bar_unit, cmap=cmap, norm=norm, spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')
    #cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm,
    #                                 spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i', **kw)
    
    cbar_width = 0.02  # Larghezza dell'asse della color bar
    cbar_height = 0.8  # Altezza dell'asse della color bar

    # Aggiungi l'asse della color bar
    cbar_ax = fig.add_axes([1.005, 0.1, cbar_width, cbar_height])  # Aggiustamento della posizione e delle dimensioni

    # Aggiungi la color bar
    cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm,
                                 spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i',  orientation='vertical', label=bar_unit)

    # Get proper ratio here
    #xmin, xmax = ax.get_xbound()
    #ymin, ymax = ax.get_ybound()
    #y2x_ratio = (ymax-ymin)/(xmax-xmin)

    #print("y2x_ratio: "+ str(y2x_ratio)) 

    #fig.set_figheight(wi * y2x_ratio + 1.)

    gs.tight_layout(fig, pad=1.2)
    fig.savefig("Grafici/" + str(fig_title) + ".png", bbox_inches="tight")
    plt.show()



def generate_extreme_maps(Slp_data, Geo_data, Slp_time, Geo_time, 
                          Most_10_extreme_dates_list, Least_10_extreme_dates_list, Xlong, Xlat, String_variable_sorted, chosen_map=precip_colormap):
    
    slp_maps_most = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    slp_maps_least = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    geo_maps_least = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    geo_maps_most = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    least_10_extreme_dates_np = [0,0,0]
    most_10_extreme_dates_np = [0,0,0]
    
    for i in range(3):
        most_10_extreme_dates_np[i]=Most_10_extreme_dates_list[i].to_numpy()  
        least_10_extreme_dates_np[i]=Least_10_extreme_dates_list[i].to_numpy() 
        #print(most_10_extreme_dates_np[i]) 
        
        for k in range(10):
            indx_m = np.where(Slp_time[:,0]==most_10_extreme_dates_np[i][k])[0][0]
            indx_l = np.where(Slp_time[:,0]==least_10_extreme_dates_np[i][k])[0][0]
            slp_maps_most[i][k] =Slp_data[indx_m]/100 # hPa
            slp_maps_least[i][k]=Slp_data[indx_l]/100 # hPa
            
            indx_gm = np.where(Geo_time[:,0]==most_10_extreme_dates_np[i][k])[0][0]
            indx_gl = np.where(Geo_time[:,0]==least_10_extreme_dates_np[i][k])[0][0] 
                # this should me equal to Slp_time but you never know
            geo_maps_most[i][k]=Geo_data[indx_gm]/g
            geo_maps_least[i][k]=Geo_data[indx_gl]/g
                     

        extreme_10_maps_contour(geo_maps_most[i], slp_maps_most[i],  Most_10_extreme_dates_list[i], Xlong, Xlat, r"Most 10 CAT" + str(i+1) + " extreme events - Z500hPa and SLP (" + str(String_variable_sorted) +  ")", bar_unit="Z [m]")
        extreme_10_maps_contour(geo_maps_least[i], slp_maps_least[i],  Least_10_extreme_dates_list[i], Xlong, Xlat,  r"Least 10 CAT" + str(i+1) + " extreme events - Z500hPa and SLP (" + str(String_variable_sorted) +  ")", bar_unit="Z [m]")
    
    
    return 1

def generate_extreme_map_composites(Slp_data, Geo_data, Slp_time, Geo_time, 
                          Most_10_extreme_dates_list, Least_10_extreme_dates_list, Xlong, Xlat, String_variable_sorted, perc=None):
    
    if perc != None:
        perc = "%"
    else:
        perc = ""
    
    slp_maps_most = [ [None for _ in range(len(Most_10_extreme_dates_list[i])) ] for i in range(3)]
    
    slp_maps_least = [ [None for _ in range(len(Least_10_extreme_dates_list[i])) ] for i in range(3)]
    

    geo_maps_most = [ [None for _ in range(len(Most_10_extreme_dates_list[i])) ] for i in range(3)]
    
    geo_maps_least = [ [None for _ in range(len(Least_10_extreme_dates_list[i])) ] for i in range(3)]

    slp_maps_most_np  = [0,0,0]
    slp_maps_least_np = [0,0,0]
    geo_maps_most_np  = [0,0,0]
    geo_maps_least_np = [0,0,0]
    
    least_10_extreme_dates_np = [0,0,0]
    most_10_extreme_dates_np = [0,0,0]
    
    for i in range(3):
        most_10_extreme_dates_np[i]=Most_10_extreme_dates_list[i].to_numpy()  
        least_10_extreme_dates_np[i]=Least_10_extreme_dates_list[i].to_numpy() 
        #print(most_10_extreme_dates_np[i]) 
        
        for k in range(len(most_10_extreme_dates_np[i])):
            print(len(slp_maps_most[i]))
            indx_m = np.where(Slp_time[:,0]==most_10_extreme_dates_np[i][k])[0][0]
            indx_l = np.where(Slp_time[:,0]==least_10_extreme_dates_np[i][k])[0][0]
            slp_maps_most[i][k] =Slp_data[indx_m]/100 # hPa
            slp_maps_least[i][k]=Slp_data[indx_l]/100 # hPa
            
            indx_gm = np.where(Geo_time[:,0]==most_10_extreme_dates_np[i][k])[0][0]
            indx_gl = np.where(Geo_time[:,0]==least_10_extreme_dates_np[i][k])[0][0] 
                # this should me equal to Slp_time but you never know
            geo_maps_most[i][k]=Geo_data[indx_gm]/g
            geo_maps_least[i][k]=Geo_data[indx_gl]/g
                     
        slp_maps_most_np[i]=np.array(slp_maps_most[i])
        slp_maps_least_np[i]=np.array(slp_maps_least[i])
        geo_maps_most_np[i]=np.array(geo_maps_most[i])
        geo_maps_least_np[i]=np.array(geo_maps_least[i])
        
        ## MOST
        # MEAN
        slp_map_mean = np.mean(slp_maps_most_np[i], axis=0)
        geo_map_mean = np.mean(geo_maps_most_np[i], axis=0)
        
        extreme_1_maps_contour(geo_map_mean, slp_map_mean, "Most 10" + perc + " CAT" + str(i+1) + " extremes mean (" + String_variable_sorted +")" , Xlong, Xlat)
        
        # STD
        # absolute
        slp_map_std = np.std(slp_maps_most_np[i], axis=0)
        geo_map_std = np.std(geo_maps_most_np[i], axis=0)    
                
        extreme_1_maps_contour_std(geo_map_std, slp_map_std, "Most 10" + perc + " CAT" + str(i+1) + " extremes std (absolute) (" + String_variable_sorted +")", Xlong, Xlat)
        # relative
        slp_map_std = slp_map_std/slp_map_mean * 100
        geo_map_std = geo_map_std/geo_map_mean * 100
        
       
        extreme_1_maps_contour_std(geo_map_std, slp_map_std, "Most 10" + perc + " CAT" + str(i+1) + " extremes std (relative %) (" + String_variable_sorted +")", Xlong, Xlat, bar_unit="Z [%]", rel=True)
        
        ## LEAST
        # MEAN
        slp_map_mean = np.mean(slp_maps_least_np[i], axis=0)
        geo_map_mean = np.mean(geo_maps_least_np[i], axis=0)
        
        extreme_1_maps_contour(geo_map_mean, slp_map_mean, "Least 10" + perc + " CAT" + str(i+1) + " extremes mean (" + String_variable_sorted +")", Xlong, Xlat)
        
        # STD
        # absolute
        slp_map_std = np.std(slp_maps_least_np[i], axis=0)
        geo_map_std = np.std(geo_maps_least_np[i], axis=0)    
                
        extreme_1_maps_contour_std(geo_map_std, slp_map_std, "Least 10" + perc + " CAT" + str(i+1) + " extremes std (absolute) (" + String_variable_sorted +")", Xlong, Xlat)
        # relative
        slp_map_std = slp_map_std/slp_map_mean * 100
        geo_map_std = geo_map_std/geo_map_mean * 100
        
        extreme_1_maps_contour_std(geo_map_std, slp_map_std, "Least 10" + perc + " CAT" + str(i+1) + " extremes std (relative %) (" + String_variable_sorted +")", Xlong, Xlat, bar_unit="Z [%]", rel=True)
        
    return 1

"""
def generate_extreme_map_composites(Slp_data, Geo_data, Slp_time, Geo_time, 
                          Most_10_extreme_dates_list, Least_10_extreme_dates_list, Xlong, Xlat, String_variable_sorted, perc=None):
    
    if perc != None:
        perc = "%"
    
    slp_maps_most = [ [None for _ in range(len(Most_10_extreme_dates_list[i])) ] for i in range(3)]
    
    slp_maps_least = [ [None for _ in range(len(Least_10_extreme_dates_list[i])) ] for i in range(3)]
    

    geo_maps_most = [ [None for _ in range(len(Most_10_extreme_dates_list[i])) ] for i in range(3)]
    
    geo_maps_least = [ [None for _ in range(len(Least_10_extreme_dates_list[i])) ] for i in range(3)]

    slp_maps_most_np  = [0,0,0]
    slp_maps_least_np = [0,0,0]
    geo_maps_most_np  = [0,0,0]
    geo_maps_least_np = [0,0,0]
    
    least_10_extreme_dates_np = [0,0,0]
    most_10_extreme_dates_np = [0,0,0]
    
    for i in range(3):
        most_10_extreme_dates_np[i]=Most_10_extreme_dates_list[i].to_numpy()  
        least_10_extreme_dates_np[i]=Least_10_extreme_dates_list[i].to_numpy() 
        #print(most_10_extreme_dates_np[i]) 
        
        for k in range(len(most_10_extreme_dates_np[i])):
            print(len(slp_maps_most[i]))
            indx_m = np.where(Slp_time[:,0]==most_10_extreme_dates_np[i][k])[0][0]
            indx_l = np.where(Slp_time[:,0]==least_10_extreme_dates_np[i][k])[0][0]
            slp_maps_most[i][k] =Slp_data[indx_m]/100 # hPa
            slp_maps_least[i][k]=Slp_data[indx_l]/100 # hPa
            
            indx_gm = np.where(Geo_time[:,0]==most_10_extreme_dates_np[i][k])[0][0]
            indx_gl = np.where(Geo_time[:,0]==least_10_extreme_dates_np[i][k])[0][0] 
                # this should me equal to Slp_time but you never know
            geo_maps_most[i][k]=Geo_data[indx_gm]/g
            geo_maps_least[i][k]=Geo_data[indx_gl]/g
                     
        slp_maps_most_np[i]=np.array(slp_maps_most[i])
        slp_maps_least_np[i]=np.array(slp_maps_least[i])
        geo_maps_most_np[i]=np.array(geo_maps_most[i])
        geo_maps_least_np[i]=np.array(geo_maps_least[i])
        
        ## MOST
        # MEAN
        slp_map_mean = np.mean(slp_maps_most_np[i], axis=0)
        geo_map_mean = np.mean(geo_maps_most_np[i], axis=0)
        
        extreme_1_maps_contour(geo_map_mean, slp_map_mean, "Most 10" + perc + " CAT" + str(i+1) + " extremes mean (" + String_variable_sorted +")" , Xlong, Xlat)
        
        # STD
        # absolute
        slp_map_std = np.std(slp_maps_most_np[i], axis=0)
        geo_map_std = np.std(geo_maps_most_np[i], axis=0)    
                
        extreme_1_maps_contour_std(geo_map_std, slp_map_std, "Most 10" + perc + " CAT" + str(i+1) + " extremes std (absolute) (" + String_variable_sorted +")", Xlong, Xlat)
        # relative
        slp_map_std = slp_map_std/slp_map_mean * 100
        geo_map_std = geo_map_std/geo_map_mean * 100
        
       
        extreme_1_maps_contour_std(geo_map_std, slp_map_std, "Most 10" + perc + " CAT" + str(i+1) + " extremes std (relative %) (" + String_variable_sorted +")", Xlong, Xlat, bar_unit="Z [%]", rel=True)
        
        ## LEAST
        # MEAN
        slp_map_mean = np.mean(slp_maps_least_np[i], axis=0)
        geo_map_mean = np.mean(geo_maps_least_np[i], axis=0)
        
        extreme_1_maps_contour(geo_map_mean, slp_map_mean, "Least 10" + perc + " CAT" + str(i+1) + " extremes mean (" + String_variable_sorted +")", Xlong, Xlat)
        
        # STD
        # absolute
        slp_map_std = np.std(slp_maps_least_np[i], axis=0)
        geo_map_std = np.std(geo_maps_least_np[i], axis=0)    
                
        extreme_1_maps_contour_std(geo_map_std, slp_map_std, "Least 10" + perc + " CAT" + str(i+1) + " extremes std (absolute) (" + String_variable_sorted +")", Xlong, Xlat)
        # relative
        slp_map_std = slp_map_std/slp_map_mean * 100
        geo_map_std = geo_map_std/geo_map_mean * 100
        
        extreme_1_maps_contour_std(geo_map_std, slp_map_std, "Least 10" + perc + " CAT" + str(i+1) + " extremes std (relative %) (" + String_variable_sorted +")", Xlong, Xlat, bar_unit="Z [%]", rel=True)
        
    return 1
"""
def generate_extreme_map_composites_all(Slp_data, Geo_data, Slp_time, Geo_time, 
                          Extremes_dates_CAT, Xlong, Xlat):
    
    slp_maps = [ [None for _ in range(len(Extremes_dates_CAT[i])) ] for i in range(3)]

    geo_maps = [ [None for _ in range(len(Extremes_dates_CAT[i])) ] for i in range(3)]

    slp_maps_np = [0,0,0]
    geo_maps_np = [0,0,0]
    
    extreme_dates_np = [0,0,0]
    
    for i in range(3):
        extreme_dates_np[i]=Extremes_dates_CAT[i].to_numpy()  
        
        for k in range(len(Extremes_dates_CAT[i])):
            indx_slp = np.where(Slp_time[:,0]==extreme_dates_np[i][k])[0][0]
            slp_maps[i][k] =Slp_data[indx_slp]/100 # hPa
            
            indx_geo = np.where(Geo_time[:,0]==extreme_dates_np[i][k])[0][0]
                # this should me equal to Slp_time but you never know
            geo_maps[i][k]=Geo_data[indx_geo]/g
                     
        slp_maps_np[i]=np.array(slp_maps[i])
        print(len(slp_maps_np[i]))
        geo_maps_np[i]=np.array(geo_maps[i])

        ## MOST
        # MEAN
        slp_map_mean = np.mean(slp_maps_np[i], axis=0)
        geo_map_mean = np.mean(geo_maps_np[i], axis=0)
        
        extreme_1_maps_contour(geo_map_mean, slp_map_mean, "All CAT" + str(i+1) + " extremes mean" , Xlong, Xlat)
        
        # STD
        # absolute
        slp_map_std = np.std(slp_maps_np[i], axis=0)
        geo_map_std = np.std(geo_maps_np[i], axis=0)    
                
        extreme_1_maps_contour_std(geo_map_std, slp_map_std, "All CAT" + str(i+1) + " extremes std (absolute)", Xlong, Xlat)
        # relative
        slp_map_std = slp_map_std/slp_map_mean * 100
        geo_map_std = geo_map_std/geo_map_mean * 100
        
       
        extreme_1_maps_contour_std(geo_map_std, slp_map_std, "All CAT" + str(i+1) + " extremes std (relative %)", Xlong, Xlat, bar_unit="Z [%]", rel=True)
        
    return 1

# COMPOSITES
def divide_dates_by_index(list_of_dates, list_of_indices, number_indices):
    # Initialize a list of empty lists
    divided_dates = [[] for _ in range(number_indices)]
    
    # Iterate through dates and indices
    for date, index in zip(list_of_dates, list_of_indices):
        # Add the date to the list corresponding to the index
        divided_dates[index].append(date)
    
    return divided_dates