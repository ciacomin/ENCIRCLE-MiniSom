# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 19:40:22 2024

@author: Criss
"""
# this code has the variables for the SOM


import numpy as np

# PATHS
path_extremes="C:/Users/Criss/Documents/Lavoro/Assegno_2024_2025/Dati/" # where you have saved the csv files
output_path ="C:/Users/Criss/Documents/Lavoro/Assegno_2024_2025/Codici/SOM/new_extremes/"

#State the path where the file is located. This will be the same path used in MiniSOM Tutorial Step #1
PATH ="C:/Users/Criss/Documents/Lavoro/Assegno_2024_2025/Codici/SOM/new_extremes/" #This is the path where the data files
folderpath = 'C:/Users/Criss/Documents/Lavoro/Assegno_2024_2025/Codici/SOM/SOMs_output/' 

#som_variables_str = "pr"
#norm_factor_CERRA_LAND = 0.2301087075443887 #pr italia

#som_variables_str = "Z500_mSLP"
#norm_factor_CERRA_LAND = 0.098014048959525 # Z500 MSLP aggiornato


#print("buh")
#print(som_variables_str)
som_variables_str = "Z500_pr"
norm_factor_CERRA_LAND =  0.098014048959525


#norm_factor_CERRA_LAND = 0.2629280725465797 # pr sicilia

#%% if you want directly to load the extremes
#extreme_list_str = "ARCIS"  
#extreme_list_str = "MSWEP"
extreme_list_str = "CERRA_LAND"
dataset_str = "CERRA_LAND"

# domain
domain_region="Italy"
#domain_region="Sicilia"
#domain_region = "Puglia"

EU_domain = "EU3"

print("Sto caricando gli estremi di "+ extreme_list_str + '_extreme_dates_' + domain_region + '.npy')
extreme_list = np.load(output_path + extreme_list_str + '_extreme_dates_' + domain_region + '.npy', allow_pickle=True)