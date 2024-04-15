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

# OPEN DATA FUNCTIONS
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

def select_latlon(ds):
    return ds.sel( lat = slice(60,36), lon = slice(-10,19)) # anche spagna e fino a scozia

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


