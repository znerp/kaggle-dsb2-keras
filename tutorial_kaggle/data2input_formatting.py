"""
# used to process the raw data and bring it into a format the neural network can use for training/predictions
# script is based on kaggle tutorial https://www.kaggle.com/kmader/mri-heart-processing 
Is closer to the original version than data2input_formatting_own.py (use of dbag which seems 
to be quite a bit faster than my implemented for loop; however, it does not save the 
patients that cannot be processed.)
"""

#%% import section
import numpy as np
import os
import sys
import psutil
from matplotlib import image
import matplotlib.pyplot as plt

# own import section
sys.path.insert(1, 'utils')
# print(sys.path)
from classes import DatasetSAX
from functions import rezoom, read_and_process

#%%
print(DatasetSAX.__doc__) # print documentation of loaded module

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
ALL_DATA_DIR =  os.path.join('C:\\Users\\Znerp\\Documents\\GitHub\\kaggle-dsb2-keras\\data', 'train', 'train')
#X_DIM, Y_DIM = 64, 64
X_DIM, Y_DIM = 128, 128
T_DIM = 30
# how many patients are processed (maximally 500)
N_PATIENTS = 3


#print(N_PATIENTS)
CD = os.getcwd()
if N_PATIENTS == 500:
    SAVENAME = os.path.join(CD, 'train_mri_{}_{}.h5'.format(X_DIM, Y_DIM))
else:
    SAVENAME = os.path.join(CD, 'train_mri_{}_{}_sub{}.h5'.format(X_DIM, Y_DIM, N_PATIENTS))
print('Savename is ' + SAVENAME + ('.\nProcessing {} patients.'.format(N_PATIENTS)))


#%% define functions etc

# # scale images to previously specified size (t,x,y)
# from scipy.ndimage import zoom
# # print(T_DIM, X_DIM, Y_DIM)
# rezoom = lambda in_data: zoom(in_data.images, [1, 
#                                                T_DIM/in_data.images.shape[1], 
#                                                X_DIM/in_data.images.shape[2], 
#                                                Y_DIM/in_data.images.shape[3]])

# continue processing
from glob import glob
base_path = os.path.join(ALL_DATA_DIR, '*')
all_series = glob(base_path) # get all paths in the folder (only directories?)

# from warnings import warn
# def read_and_process(in_path): # reads data in with pre-defined function and scales it to 128x128
#     try:
#         cur_data = DatasetSAX(in_path,
#                            os.path.basename(in_path)) # gets the name of the lowest folder
#         cur_data.load()
#         if cur_data.time is not None: # when would that ever be none? only if no sax data was found? but then there would be a problem anyway!?
#             zoom_time = zoom(cur_data.time, [T_DIM/len(cur_data.time)]) 
#         else:
#             zoom_time = range(T_DIM)
#         return [in_path, zoom_time, cur_data.area_multiplier, rezoom(cur_data)] # scale images
#     except Exception as e: # catches exceptions without letting them stop the code
#         warn('\nWarning: {}\nPatient: {}'.format(e, os.path.basename(in_path)), RuntimeWarning)
#         failedPatients.append()
#         return None

# read and process one example
a,d,b,c = read_and_process(all_series[-100])
print(c.shape)

## Processing of MULTIPLE examples
import dask
import dask.diagnostics as diag
from bokeh.io import output_notebook
from bokeh.resources import CDN
from dask import bag as dbag
from multiprocessing.pool import ThreadPool

path_bag = dbag.from_sequence(np.random.choice(all_series, N_PATIENTS)) # randomly selects N patients
image_bag = path_bag.map(read_and_process) # maps the function to all elements of the sequence

#%% computation
with diag.ProgressBar(), diag.Profiler() as prof, diag.ResourceProfiler(0.5) as rprof, dask.config.set(pool = ThreadPool(4)):
    all_img_data = image_bag.compute()

#%% 
print(len(all_img_data))

#%%
im_stack = np.concatenate([x[-1] for x in all_img_data if x is not None],0)
print(im_stack.shape)

#%%
# area multiplier stack
am_stack = np.concatenate([ [x[2]]*x[-1].shape[0] for x in all_img_data if x is not None],0)
print(am_stack.shape)

#%%
# id stack
path_stack = np.concatenate([ [os.path.basename(x[0])]*x[-1].shape[0] for x in all_img_data if x is not None],0)
print(path_stack.shape)

#%%
time_stack = np.concatenate([ [x[1]]*x[-1].shape[0] for x in all_img_data if x is not None],0)
print(time_stack.shape)

#%%
import pandas as pd
train_file = 'C:\\Users\\Znerp\\Documents\\GitHub\\kaggle-dsb2-keras\\data\\train.csv'
train_targets = {int(k['Id']): k for k in pd.read_csv(train_file).T.to_dict().values()}

#%% 
os.getcwd()
#%% save
import h5py
with h5py.File(SAVENAME, 'w') as w:
    w.create_dataset('image', data = im_stack, compression = 9)
    w.create_dataset('systole', data = [train_targets[int(c_id)]['Systole'] for c_id in path_stack])
    w.create_dataset('diastole', data = [train_targets[int(c_id)]['Diastole'] for c_id in path_stack])
    w.create_dataset('id', data = [int(c_id) for c_id in path_stack])
    w.create_dataset('area_multiplier', data = am_stack)
    w.create_dataset('time', data = time_stack)