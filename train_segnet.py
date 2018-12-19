### 17.12.2018
# Training modelv03 (single slice SegNet = 3-SN) with hand labeled data from Julian de Wit
# (Sunnybrook dataset not yet accessible)
import os
import time
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import json
import csv

import sys
sys.path.insert(1, 'C:\\Users\\Znerp\\Documents\\GitHub\\kaggle-dsb2-keras\\model')
from modelv03 import segnet_single_slice
from git_utils import crps, real_to_cdf, preprocess, rotation_augmentation, shift_augmentation, split_data

data_folder = 'C:\\Users\\Znerp\\Documents\\GitHub\\kaggle-dsb2-keras\\data\\segmentation'
save_folder = 'C:\\Users\\Znerp\\Documents\\GitHub\\kaggle-dsb2-keras\\training_results\\segmentation\\' + time.strftime('%Y%m%d_%H%M')

# Hyperparameters (implement in smart way)
# specify loss, optimizer and metric further down
hypers = {'model': 'segnet_single_slice',
'pre-processing': False,
'post-processing': False,
'data_augment': False,
'keep_prob': 1,
'nb_iter': 500, 
'epochs_per_iter': 1, 
'batch_size': 32, # memory of GPU too low for 128? 
'calc_crps': 0,
'm': 2500, # number of training examples trained on (11319 total)
'rnd_seed': 176,
'image_size': [184,184]} 



print('Loading training data...')
filenames = [ os.path.join(data_folder,'X_train_{}.npy'.format(hypers['image_size'][0])),
                os.path.join(data_folder,'y_train_{}.npy'.format(hypers['image_size'][0])) ]
X = np.load(filenames[0])
y = np.load(filenames[1])

train_set_size = X.shape[0]
print(str(train_set_size) + ' total training examples loaded.')

# pick subset of training data as specified above by m
np.random.seed(hypers['rnd_seed']) # set seed for replicable results
trainIDs = np.random.choice(train_set_size, hypers['m'])

X_train = X[trainIDs,:,:]
y_train = y[trainIDs,:,:]
print('Shape of trainings examples is:' + str(X.shape[1:]))


print('Loading and compiling models...')
# would like to save the optimizer data differently, but do not know how
loss_fct = 'binary_crossentropy'
optim_params = {'lr': 9e-4, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': None, 'decay': 0.0, 'amsgrad': False}
optim = optimizers.Adam(**optim_params)#(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

hypers['loss'] = loss_fct
hypers['optimizer'] = {'name': 'Adam', 'params': optim_params} # stack dictionaries

SSSN = segnet_single_slice(input_shape=X.shape[1:], keep_prob=hypers['keep_prob'])
SSSN.compile(optimizer = optim, loss = loss_fct)


if hypers['pre-processing'] != False: # no pre-processing for now
    print('Pre-processing images...')
    X = preprocess(X) # denoise_tv_chambolle

# no split into training and test

# create a new folder named after current date and time to save all model parameters and infos regarding the training process
os.makedirs(save_folder)
# save losses and hyperparameters in json file
metadict = {'hparms': hypers, 'losses': None, 'duration': None}
with open(os.path.join(save_folder, 'metadata.json'), 'w') as jf:
    json.dump(metadict, jf)

### Actual training
t_start = time.time()

# remember min val. losses 
min_val_loss = sys.float_info.max #############################################

print('-'*50)
print('Training...')
print('-'*50)

# store losses (training and validation; systole and diastole) on every iteration to evaluate learning progress 
losses = []

for i in range(hypers['nb_iter']):
    print('-'*50)
    print('Iteration {0}/{1}'.format(i + 1, hypers['nb_iter']))
    print('-'*50)

    if hypers['data_augment'] != False: # would have to be changed if desired
        print('Augmenting images - rotations')
        X_train = rotation_augmentation(X_train, 15)
        print('Augmenting images - shifts')
        X_train = shift_augmentation(X_train, 0.1, 0.1)

    print('Fitting segmentation model...')
    hist = SSSN.fit(X_train, y_train, batch_size=hypers['batch_size'], epochs=hypers['epochs_per_iter'], verbose = 2) 

    # sigmas for predicted data, actually loss function values (RMSE)
    loss_value = hist.history['loss'][-1]
    # print(hist.history, loss_value)

    # write losses instead of storing them in case code execution has to be stopped
    with open(os.path.join(save_folder, 'val_loss_all.txt'), mode='a') as f:
        # f = csv.reader(csvf, lineseparator='\n')
        f.write(str(loss_value) + '\n')

    # store loss values for: train, val <-> systole, diastole
    losses.append(loss_value)


    # for best (lowest) val losses, save weights
    if loss_value < min_val_loss:
        print('Loss reduced. Saving weights...')
        min_val_loss = loss_value
        SSSN.save_weights(os.path.join(save_folder, 'weights_best.hdf5'), overwrite=True)


t_end = time.time()
t_delta = t_end-t_start

print('Done with training the CNN with {} images. Total time elapsed is {} min.'.format(hypers['m'], t_delta/60))

# append losses and duration to metadict
metadict['losses'] = losses
metadict['duration'] = t_delta
metadict['hparms']['loss_best'] = np.min(losses)

with open(os.path.join(save_folder, 'metadata.json'), 'w+') as jf:
    json.dump(metadict, jf)




