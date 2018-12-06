#%%
import numpy as np
import os
import sys
from matplotlib import image
import matplotlib.pyplot as plt

pathToUtils = 'C:\\Users\\Znerp\\Documents\\GitHub\\kaggle-dsb2-keras\\tutorial_kaggle\\utils'
sys.path.insert(1, pathToUtils)
from classes import DatasetSAX


#%% useful variables
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
X_DIM, Y_DIM = 64, 64
X_DIM, Y_DIM = 128, 128
T_DIM = 30

# %% Loading the data of one random patient
np.random.seed(17)
num_patients = len(os.listdir(ALL_DATA_DIR))
# patientID = str(np.random.randint(1,num_patients+1)) # get random patient
patientID = '41'
base_path = os.path.join(ALL_DATA_DIR, patientID)
print(base_path)

#%%
tData = DatasetSAX(base_path, patientID)
# print(tData)
print(tData.directory)
print(tData.folders)
print(tData.slices)

#%% load patient data
tData.load()

#%% Quick overview over the parameters
attribute_dict = vars(tData)
attributes = attribute_dict.keys()
print(attributes)

for key,val in attribute_dict.items():
    if key != 'images':
        print(key, val)

#%% Visualization
print('First image of {} images for patient {}: '.format(tData.images.shape[0]*tData.images.shape[1], tData.name))
plt.imshow(tData.images[0,0,:,:], cmap = 'bone')

