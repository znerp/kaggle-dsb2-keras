# 19.12.2018
# prediction of segmentation based on validation dataset

#%%
import h5py
import numpy as np
from keras import optimizers
import sys, os
sys.path.insert(1, 'C:\\Users\\Znerp\\Documents\\GitHub\\kaggle-dsb2-keras\\model')
from modelv03 import segnet_single_slice

#%% some declarations
weight_folder = 'C:\\Users\\Znerp\\Documents\\GitHub\\kaggle-dsb2-keras\\training_results\\segmentation\\20181219_0948'
data_file = 'C:\\Users\\Znerp\\Documents\\GitHub\\kaggle-dsb2-keras\\data\\train\\train_mri_128_128_N500.h5'
# data_file = 'C:\\Users\\Znerp\\Documents\\GitHub\\kaggle-dsb2-keras\\data\\validate\\validate_mri_64_64_N200.h5'
save_folder = 'C:\\Users\\Znerp\\Documents\\GitHub\\kaggle-dsb2-keras\\predictions'

save_pred = True

nImages = 512 # 61290 in total in validation set, 158220 in training set
batch_size = 32
loss_fct = 'binary_crossentropy'
optim_params = {'lr': 9e-4, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': None, 'decay': 0.0, 'amsgrad': False}
optim = optimizers.Adam(**optim_params)

print('Loading data.')
with h5py.File(data_file, 'r') as h5f:
    X = h5f['image'].value
    ids = h5f['id'].value
print('Shape of images is ' + str(X.shape))
# adjust input size to the one required by the single slice segmentation model 
X_ss = X.reshape((X.shape[0]*X.shape[1], 1, X.shape[2], X.shape[3]))
# ids_ss = np.tile(ids, (30, 1)) # create 30 copies of the ids
# ids_ss = np.rollaxis(ids_ss, 1, 0) # change the dimensions, s.t. the order is correct after reshaping
# ids_ss = ids_ss.reshape((np.multiply(*ids_ss.shape)))
print('--> ' + str(X_ss.shape[0]) + ' total images.')

print('Initializing model.')
input_shape = (1,*X.shape[2:])
SSSN = segnet_single_slice(input_shape)
SSSN.compile(optimizer=optim, loss=loss_fct)
SSSN.load_weights(os.path.join(weight_folder, 'weights_best.hdf5'))
# print(SSSN.summary())

print('Predicting {} images.'.format(nImages))
predictions = SSSN.predict(x = X_ss[:nImages,:,:,:], batch_size=batch_size, verbose=1)
# convert predictions to only 0s and 1s
predictions[predictions<=0.5] = 0
predictions[predictions>0.5] = 1


#%% predictions are all zero...
if save_pred: # for looking at the predictions
    print('Saving predicted binary images and originals.')
    import scipy.misc
    for i in range(nImages):
        j = int(i/30)
        savename_base = os.path.join(save_folder, '{:04d}_{:03d}.png'.format(ids[j], i+1)) # i should be exchanged for a counter per patient
        savename_out = savename_base[:-4] + '-o.png'
        # print(savename_base, savename_out)

        scipy.misc.imsave(savename_base, X_ss[i,:,:,:].squeeze(axis=0))
        scipy.misc.imsave(savename_out, predictions[i,:,:,:].squeeze(axis=0))


# # for every slice, calculate area from images and get mimimum and maximum value (diastole and systole) 
# counter = 0

# systole = []
# diastole = []
# for i in len(ids):
#     for j in range(counter*30, (counter+1)*30-1):

#     counter += 1