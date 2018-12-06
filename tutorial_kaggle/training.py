import os
import h5py
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
import numpy as np
import csv
# import gc
# gc.enable() # we come close to the memory limits and this seems to minimize kernel resets

data_dir = 'C:\\Users\\Znerp\\Documents\\GitHub\\kaggle-dsb2-keras\\data'
filename = os.path.join(data_dir, 'train_mri_64_64_sub5.h5')
# filename = os.path.join(data_dir,'train_mri_128_128.h5')

# fily = h5py.File(filename, 'r')

with h5py.File(filename, 'r') as w:
    full_data = w['image'].value
    n_group = w['id'].value
    n_scalar = w['area_multiplier'].value
    y_target_sys = w['systole'].value / n_scalar # remove the area scalar since we dont have this in the images
    y_target_dia = w['diastole'].value / n_scalar 

print(full_data[0].shape)


