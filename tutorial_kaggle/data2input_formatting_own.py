#%% import section
import numpy as np
import os
import sys
import psutil
from matplotlib import image
import matplotlib.pyplot as plt
import argparse


pathToUtils = 'C:\\Users\\Znerp\\Documents\\GitHub\\kaggle-dsb2-keras\\tutorial_kaggle\\utils'
sys.path.insert(1, pathToUtils)
from classes import DatasetSAX
from functions import rezoom, read_and_process

#%%
# print(read_and_process.__doc__) # print documentation of loaded module

#%%
# number of bins to use in histogram for gaussian regression
NUM_BINS = 100
# number of standard deviatons past which we will consider a pixel an outlier
STD_MULTIPLIER = 2
# number of points of our interpolated dataset to consider when searching for
# a threshold value; the function by default is interpolated over 1000 points,
# so 250 will look at the half of the points that is centered around the known
# myocardium pixel
THRESHOLD_AREA = 250
# number of pixels on the line within which to search for a connected component
# in a thresholded image, increase this to look for components further away
COMPONENT_INDEX_TOLERANCE = 20
# number of angles to search when looking for the correct orientation
ANGLE_SLICES = 36
X_DIM, Y_DIM = 64, 64
# X_DIM, Y_DIM = 128, 128
T_DIM = 30
# how many patients are processed (maximally 500)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', dest='n', type=int, default=500)
    args = parser.parse_args()

    N_PATIENTS = args.n
else:
    N_PATIENTS = 500

DATA_DIR = 'C:\\Users\\Znerp\\Documents\\GitHub\\kaggle-dsb2-keras\\data'
ALL_PATIENTS_DIR =  os.path.join(DATA_DIR, 'train', 'train')
TRAIN_FILE = 'C:\\Users\\Znerp\\Documents\\GitHub\\kaggle-dsb2-keras\\data\\train.csv'


#print(N_PATIENTS)
CD = os.getcwd()
if N_PATIENTS == 500:
    SAVENAME = os.path.join(DATA_DIR, 'train_mri_{}_{}.h5'.format(X_DIM, Y_DIM))
else:
    SAVENAME = os.path.join(DATA_DIR, 'train_mri_{}_{}_sub{}.h5'.format(X_DIM, Y_DIM, N_PATIENTS))
print('Savename is ' + SAVENAME + ('.\nProcessing {} patients.'.format(N_PATIENTS)))

#%%
# print(np.random.choice.__doc__)
# print(np.random.__doc__)

# #%% continue processing
# from glob import glob
# base_path = os.path.join(ALL_PATIENTS_DIR, '*')
# all_series = glob(base_path) # get all paths in the folder (only directories?)
# print(all_series[:10])

# #%% read and process one example
# a,d,b,c = read_and_process(all_series[-100])
# print(c.shape)

#%% Processing of MULTIPLE examples
# from keras.utils.generic_utils import Progbar
from warnings import warn
from time import time

all_patient_dirs = os.listdir(ALL_PATIENTS_DIR)
# progbar = Progbar(len(base_path))
np.random.seed(17)
sel_patients = np.random.choice(np.arange(1,501), N_PATIENTS)

all_series = []
failedPatients = []
it=1
print('Starting loop for data read-in of {0} patients at image resolution of {1}x{1} pixels.'.format(N_PATIENTS, X_DIM))
start = time()
for patient in sel_patients: # process patient unless error; in this case save patient and error message
    patient_dir = os.path.join(ALL_PATIENTS_DIR, str(patient))
    # print(patient_dir)
    try:
        a,b,c,d = read_and_process(patient_dir, t_dim=T_DIM, x_dim=X_DIM, y_dim=Y_DIM, withErrCatch=False)
        all_series.append([a,b,c,d])
    except Exception as e: # catches exceptions without letting them stop the code
        # warn('\nPatient {}, warning:{}'.format(patient, e), RuntimeWarning)
        print('Patient {}, warning:{}'.format(patient, e))
        failedPatients.append((patient, e))
    end = time()
    print('Iteration {}, patient {} done after {:.2f} seconds. Presumably {:.2f} seconds remaining.'.format(it, patient, end-start, (end-start)/it*(N_PATIENTS-it)))
    it += 1

# print(failedPatients)
#     # progbar.add(1)

#%%
# print(all_series[0])
all_img_data = all_series

#%% 
#print(all_img_data[all_img_data == None])
# print(len(all_img_data[0]))
# print(all_img_data[0][3].shape)

#%%
im_stack = np.concatenate([x[-1] for x in all_img_data if x is not None],0)
print('Shape of image stack: ' + str(im_stack.shape))

#%%
# area multiplier stack
am_stack = np.concatenate([ [x[2]]*x[-1].shape[0] for x in all_img_data if x is not None],0)
print('Shape of area stack: ' + str(am_stack.shape))

#%%
# id stack
path_stack = np.concatenate([ [os.path.basename(x[0])]*x[-1].shape[0] for x in all_img_data if x is not None],0)
print('Shape of path stack: ' + str(path_stack.shape))

#%%
time_stack = np.concatenate([ [x[1]]*x[-1].shape[0] for x in all_img_data if x is not None],0)
print('Shape of time stack: ' + str(time_stack.shape))

#%%
import pandas as pd
train_file = 'C:\\Users\\Znerp\\Documents\\GitHub\\kaggle-dsb2-keras\\data\\train.csv'
train_targets = {int(k['Id']): k for k in pd.read_csv(train_file).T.to_dict().values()}

#%% save
import h5py
print('Saving processed data...')
with h5py.File(SAVENAME, 'w') as w:
    w.create_dataset('image', data = im_stack, compression = 9)
    w.create_dataset('systole', data = [train_targets[int(c_id)]['Systole'] for c_id in path_stack])
    w.create_dataset('diastole', data = [train_targets[int(c_id)]['Diastole'] for c_id in path_stack])
    w.create_dataset('id', data = [int(c_id) for c_id in path_stack])
    w.create_dataset('area_multiplier', data = am_stack)
    w.create_dataset('time', data = time_stack)
print('Done.')

#%% 
import csv
print('Saving erroneous patients...')
with open(os.path.join(DATA_DIR, 'failed.csv'), 'w') as csvfile:
    for item in failedPatients:
        # print(str(item))
        csvfile.write(str(item) + '\n')
print('Done.')
#%%
