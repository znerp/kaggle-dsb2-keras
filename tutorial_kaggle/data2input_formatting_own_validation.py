### used to process the raw data and bring it into a format the neural network can use for training/predictions
# script is based on kaggle tutorial https://www.kaggle.com/kmader/mri-heart-processing 

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


#%% some metavariables
X_DIM, Y_DIM = 64, 64
# X_DIM, Y_DIM = 128, 128
T_DIM = 30
# how many patients are processed (maximally 200 for validation)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', dest='n', type=int, default=200)
    args = parser.parse_args()

    N_PATIENTS = args.n
else:
    N_PATIENTS = 200

DATA_DIR = 'C:\\Users\\Znerp\\Documents\\GitHub\\kaggle-dsb2-keras\\data'
ALL_PATIENTS_DIR =  os.path.join(DATA_DIR, 'validate', 'validate') 


# checks for file overwrite
SAVENAME = os.path.join(DATA_DIR, 'validate', 'validate_mri_{}_{}_N{}.h5'.format(X_DIM, Y_DIM, N_PATIENTS))
if os.path.isfile(SAVENAME):
    print('Savename is ' + SAVENAME + (' and {}.\nProcessing {} patients.'.format('already exists.', N_PATIENTS)))
else: 
    print('Savename is ' + SAVENAME + (' and {}.\nProcessing {} patients.'.format('does not exist yet.', N_PATIENTS)))

fail_name = os.path.join(DATA_DIR, 'validate', 'failedPatients_validate_N{}.csv'.format(N_PATIENTS))
if os.path.isfile(fail_name):
    overwrite = input('Code execution would lead to overwrite of  {}. Shoult it really be overwritten?\n [y/n] '.format(fail_name))
    if overwrite != 'y':
        raise Exception('Files will not be overwritten.')




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
sel_patients = np.random.choice(np.arange(501,701), N_PATIENTS)

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


#%% save
import h5py
print('Saving processed data...')
with h5py.File(SAVENAME, 'w') as w:
    w.create_dataset('image', data = im_stack, compression = 9)
    w.create_dataset('id', data = [int(c_id) for c_id in path_stack])
    w.create_dataset('area_multiplier', data = am_stack)
    w.create_dataset('time', data = time_stack)
print('Done.')

#%% save failed patients
import csv
print('Saving erroneous patients...')

with open(fail_name, 'w') as csvfile:
    for item in failedPatients:
        # print(str(item))
        csvfile.write(str(item) + '\n')
print('Done.')
