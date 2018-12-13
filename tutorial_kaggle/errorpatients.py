### 06.12.2018
# Script for getting more information about the patients that could not be properly read in by the DatasetSAX class


# #%% to understand os.walk()
# import os
# for root, dirs, files in os.walk(os.getcwd()):
#     print('\nRoot is:', root)
#     for name in files:
#         print('File:', os.path.join(root, name))
#     for name in dirs:
#         print('Dir:', os.path.join(root, name))


#%%
import os
data_dir = 'C:\\Users\\Znerp\\Documents\\GitHub\\kaggle-dsb2-keras\\data\\train'

# # if file was actually saved with a delimiter between patients
# with open(os.path.join(data_dir,'failed.csv'), 'r') as csvfile:
#     content = csv.reader(csvfile)
#     for row in content:
#         failedPatients.append(row)

# for the current way where all patients and warnings are just one string
with open(os.path.join(data_dir, 'failedPatients_train_N500.csv'), 'r') as csvfile:
    failed = csvfile.readlines()

nFailedPatients = len(failed)

#%% manually go through patients whose processing step failed
import sys
sys.path.insert(1, 'C:\\Users\\Znerp\\Documents\\GitHub\\kaggle-dsb2-keras\\tutorial_kaggle\\utils')
from classes import DatasetSAX
from functions import read_and_process

pIDs = ['279', '442', '41', '416', '456']
pPath = os.path.join(data_dir, 'train', pIDs[-1])
# pData = DatasetSAX(pPath, pID)
# pData.load()
read_and_process(pPath)

#%% processing of these patients occurs without error
pnIDs = ['1', '2', '3']
pnPath = os.path.join(data_dir, 'train', pnIDs[0])
# pData = DatasetSAX(pPath, pID)
# pData.load()
# read_and_process(pnPath)


#%% look at metadata of files of one patient
import pydicom as dicom
import re

def read_dicom_1patient(patient_dir):
    """
    Reads in all dicom images in the SAX folders of one patient.

    Input:
    patient_dir: path to the directory of the patient

    Returns:
    meta: List of metadata of all dicom files
    nFolders: number of folders processed
    """
    directory = patient_dir

    while True:
        subdirs = next(os.walk(directory))[1]
        if len(subdirs) == 1:
            directory = os.path.join(directory, subdirs[0])
        else:
            break

    meta = []
    nFolders = 0
    for s in subdirs: # gets the SAX slices of the patient
        m = re.match("sax_(\d+)", s)
        if m is not None: # --> pattern is matched
            nFolders += 1
            files = next(os.walk(os.path.join(directory, s)))[2] # gets filenames in sax folder; os.walk -> (dirpath, dirnames, filenames)
            for f in files:
                d = dicom.read_file(os.path.join(directory, s, f))
                meta.append(d)

    return meta, nFolders

meta, nFolders = read_dicom_1patient(pPath)

#%%
print(len(meta))
print(nFolders)

slic = 0
for key in meta[slic].keys():
    print(meta[slic][key])

#%% save one example of metadata
if False: # to prevent saving every time
    print(os.getcwd())
    with open('metadata_ex_{}.txt'.format(pnPath[-1:]), 'w') as f:
        f.write('Metadata:\n')
        for key in meta[slic].keys():
            f.write(str(meta[slic][key]))
            f.write('\n')

#%%
for slic in range(len(meta)):
    print(meta[slic].pixel_array.shape)

# img = meta[slic].pixel_array
# print(img.shape)
# print(meta[slic].PatientID)

#%%
