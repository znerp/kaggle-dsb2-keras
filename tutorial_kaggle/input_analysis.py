########## Input analysis for number of folders and distribution into 2ch, 4ch, sax etc.
#%%
import sys
import os
import numpy as np

pathToUtils = 'C:\\Users\\Znerp\\Documents\\GitHub\\kaggle-dsb2-keras\\tutorial_kaggle\\utils'
sys.path.insert(1, pathToUtils)
from classes import DatasetSAX

TRAIN_DIR = os.path.join('C:\\Users\\Znerp\\Documents\\GitHub\\kaggle-dsb2-keras\\data', 'train', 'train')
VAL_DIR = os.path.join('C:\\Users\\Znerp\\Documents\\GitHub\\kaggle-dsb2-keras\\data', 'validate', 'validate')
ALL_PATIENTS_DIR = TRAIN_DIR


#%%
nPatients = len(os.listdir(ALL_PATIENTS_DIR))
pData = []

# # with paths (allowing skips in patient numbers)
# for patient_dir in os.listdir(ALL_PATIENTS_DIR):
#     # print(patient_dir, os.path.basename(patient_dir))
#     # print(ALL_PATIENTS_DIR, patient_dir)
#     pData.append(DatasetSAX(os.path.join(ALL_PATIENTS_DIR, patient_dir), patient_dir)) # create object of class DatasetSAX

# with patient numbers (assuming that every patient is listed as is the case in 
# the training data) -> results in correct order 
for n in range(nPatients):
    patient_dir = str(n+1)
    pData.append(DatasetSAX(os.path.join(ALL_PATIENTS_DIR, patient_dir), patient_dir)) # create object of class DatasetSAX
    
#%% assembling of the number of folder
# # more sophisticated version if the order is incorrect
# folder_map = {'{:03d}'.format(int(x.name)): x.folders for x in pData}
# num_folders = {key: len(val) for key,val in folder_map.items()}

# for correct order of the patients (and no missing patients)
folder_map = [ds.folders for ds in pData]
num_folders = {'total': np.asarray([len(ds) for ds in folder_map])}


#%%
print(num_folders)
print(folder_map)

#%% more into detail: how many sax folders vs 2ch vs 4ch images
import re
sax_folder_map = [[folder for folder in patient if (re.match('sax', folder)!= None)] for patient in folder_map]
ch2_folder_map = [[folder for folder in patient if (re.match('2ch', folder)!= None)] for patient in folder_map]
ch4_folder_map = [[folder for folder in patient if (re.match('4ch', folder)!= None)] for patient in folder_map]
print(ch4_folder_map)

#%% find min and max number of folders
num_folders['sax'] = np.asarray([len(ds) for ds in sax_folder_map])
num_folders['2ch'] = np.asarray([len(ds) for ds in ch2_folder_map])
num_folders['4ch'] = np.asarray([len(ds) for ds in ch4_folder_map])

#%%
print(num_folders.keys())
print(np.sum(num_folders['2ch'])) # all patients got exactly one 2chamber-view!

#%% min and max
avg = {}
mini = {}
maxi = {}
for key in num_folders.keys():
    avg[key] = np.average(num_folders[key])
    mini[key] = np.min(num_folders[key])
    maxi[key] = np.max(num_folders[key])

#%%
print('The {} patients have...'.format(nPatients))
for key in num_folders.keys():
    print('...on average {0} {3} folders, with a minimum of {1} and a maximum of {2} {3} folders.'.format(avg[key], mini[key], maxi[key], key))