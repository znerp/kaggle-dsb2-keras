### 12.12.2018
# Training a neural network on data processed in a way inspired by the kaggle tutorial
# Now with model v02 (flexible input shape) and saving of more parameters
# without pre-processing and data augmentation -> making it as simple as possible
import os
import time
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
# import h5py
import json

import sys
sys.path.insert(1, 'C:\\Users\\Znerp\\Documents\\GitHub\\kaggle-dsb2-keras\\model')
from modelv02 import get_model, RMSE
from git_utils import crps, real_to_cdf, preprocess, rotation_augmentation, shift_augmentation, load_train_data, split_data

data_folder = 'C:\\Users\\Znerp\\Documents\\GitHub\\kaggle-dsb2-keras\\data\\train'
save_folder = 'C:\\Users\\Znerp\\Documents\\GitHub\\kaggle-dsb2-keras\\training_results\\' + time.strftime('%Y%m%d_%H%M')

# Hyperparameters (implement in smart way)
# specify loss, optimizer and metric further down
hypers = {'model': 'modelv02',
'pre-processing': False,
'post-processing': False,
'data_augment': False,
'keep_prob': 1,
'nb_iter': 400, 
'epochs_per_iter': 1, 
'batch_size': 64, # memory of GPU too low for 256? 
'calc_crps': 0,
'm': 500, # number of training examples trained on
'rnd_seed': 176
'image_size': [64,64]} 



print('Loading training data...')
filename = os.path.join(data_folder,'train_mri_{0}_{1}_N500.h5'.format(*hypers['image_size']))
X, y = load_train_data(filename)
y = y.T

train_set_size = X.shape[0]
print(str(train_set_size) + 'total training examples loaded.')

# pick subset of training data as specified above by m
np.random.seed(hypers['rnd_seed']) # set seed for replicable results

trainIDs = np.random.choice(train_set_size, hypers['m'])
X = X[trainIDs,:,:,:]
y = y[trainIDs,:]
print('Shape of trainings examples is:' + str(X.shape[1:]))


print('Loading and compiling models...')
# would like to save the optimizer data differently, but do not know how
loss_fct = RMSE
optim_params = {'lr': 7.5e-4, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': None, 'decay': 0.0, 'amsgrad': False}
optim = optimizers.Adam(**optim_params)#(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

hypers['loss'] = loss_fct.__name__
hypers['optimizer'] = {'name': 'Adam', 'params': optim_params} # stack dictionaries

model_systole = get_model(input_shape=X.shape[1:], keep_prob=hypers['keep_prob'])
model_systole.compile(optimizer = optim, loss = loss_fct)
model_diastole = get_model(input_shape=X.shape[1:], keep_prob=hypers['keep_prob'])
model_diastole.compile(optimizer = optim, loss = loss_fct)


if hypers['pre-processing'] != False: # no pre-processing for now
    print('Pre-processing images...')
    X = preprocess(X) # denoise_tv_chambolle

# split to training and test
X_train, y_train, X_test, y_test = split_data(X, y, split_ratio=0.2)


### Actual training
t_start = time.time()

# remember min val. losses (best iterations), used as sigmas for submission
min_val_loss_systole = sys.float_info.max
min_val_loss_diastole = sys.float_info.max

print('-'*50)
print('Training...')
print('-'*50)

# store losses (training and validation; systole and diastole) on every iteration to evaluate learning progress 
loss = {'train': [], 'val': []}

for i in range(hypers['nb_iter']):
    print('-'*50)
    print('Iteration {0}/{1}'.format(i + 1, hypers['nb_iter']))
    print('-'*50)

    if hypers['data_augment'] != False:
        print('Augmenting images - rotations')
        X_train = rotation_augmentation(X_train, 15)
        print('Augmenting images - shifts')
        X_train = shift_augmentation(X_train, 0.1, 0.1)

    print('Fitting systole model...')
    hist_systole = model_systole.fit(X_train, y_train[:, 0], batch_size=hypers['batch_size'], epochs=hypers['epochs_per_iter'], verbose = 2,
                                        validation_data=(X_test, y_test[:,0]), shuffle = True) 

    print('Fitting diastole model...')
    hist_diastole = model_diastole.fit(X_train, y_train[:, 1], batch_size=hypers['batch_size'], epochs=hypers['epochs_per_iter'], verbose = 2,
                                        validation_data=(X_test, y_test[:,1]), shuffle = True) 

    # sigmas for predicted data, actually loss function values (RMSE)
    loss_systole = hist_systole.history['loss'][-1]
    loss_diastole = hist_diastole.history['loss'][-1]
    val_loss_systole = hist_systole.history['val_loss'][-1]
    val_loss_diastole = hist_diastole.history['val_loss'][-1]

    # store loss values for: train, val <-> systole, diastole
    loss['train'].append([loss_systole, loss_diastole])
    loss['val'].append([val_loss_systole, val_loss_diastole])

    if hypers['calc_crps'] > 0 and i % hypers['calc_crps'] == 0:
        print('Evaluating CRPS...')
        pred_systole = model_systole.predict(X_train, batch_size=hypers['batch_size'], verbose=1)
        pred_diastole = model_diastole.predict(X_train, batch_size=hypers['batch_size'], verbose=1)
        val_pred_systole = model_systole.predict(X_test, batch_size=hypers['batch_size'], verbose=1)
        val_pred_diastole = model_diastole.predict(X_test, batch_size=hypers['batch_size'], verbose=1)

        # CDF for train and test data (actually a step function)
        cdf_train = real_to_cdf(np.concatenate((y_train[:, 0], y_train[:, 1])))
        cdf_test = real_to_cdf(np.concatenate((y_test[:, 0], y_test[:, 1])))

        # CDF for predicted data
        cdf_pred_systole = real_to_cdf(pred_systole, loss_systole)
        cdf_pred_diastole = real_to_cdf(pred_diastole, loss_diastole)
        cdf_val_pred_systole = real_to_cdf(val_pred_systole, val_loss_systole)
        cdf_val_pred_diastole = real_to_cdf(val_pred_diastole, val_loss_diastole)

        # evaluate CRPS on training data
        crps_train = crps(cdf_train, np.concatenate((cdf_pred_systole, cdf_pred_diastole)))
        print('CRPS(train) = {0}'.format(crps_train))

        # evaluate CRPS on test data
        crps_test = crps(cdf_test, np.concatenate((cdf_val_pred_systole, cdf_val_pred_diastole)))
        print('CRPS(test) = {0}'.format(crps_test))

    
    # create a new folder named after current date and time to save all model parameters and infos regarding the training process
    if i == 0:
        os.makedirs(save_folder)

    # for best (lowest) val losses, save weights
    if val_loss_systole < min_val_loss_systole:
        print('Systole loss reduced. Saving weights...')
        min_val_loss_systole = val_loss_systole
        model_systole.save_weights(os.path.join(save_folder, 'weights_systole_best.hdf5'), overwrite=True)

    if val_loss_diastole < min_val_loss_diastole:
        print('Diastole loss reduced. Saving weights...')
        min_val_loss_diastole = val_loss_diastole
        model_diastole.save_weights(os.path.join(save_folder, 'weights_diastole_best.hdf5'), overwrite=True)


t_end = time.time()
t_delta = t_end-t_start

print('Done with training the CNN with {} sets of images. Total time elapsed is {} min.'.format(train_set_size, t_delta/60))


print('*'*50)
print('Saving Losses and Hyperparameters.')
print('*'*50)

# save best (lowest) val losses in readable file (to be later used for generating submission)
with open(os.path.join(save_folder, 'val_loss_all.txt'), mode='w+') as f:
    f.write('train-sys\ttrain-dia\tval-sys\tval-dia\n')
    for i in range(hypers['nb_iter']):
        f.write('{}\t{}\t{}\t{}\n'.format(loss['train'][i][0], loss['train'][i][1], loss['val'][i][0], loss['val'][i][1]))

# save hyperparameters to readable file
with open(os.path.join(save_folder, 'hyperparameters.txt'), mode='w+') as g:
    for key,val in hypers.items():
        g.write('{}: {}\n'.format(key, val))

# save losses and hyperparameters in json file
metadict = {'hparms': hypers, 'losses': loss, 'duration': t_delta}
with open(os.path.join(save_folder, 'metadata.json'), 'w') as jf:
    json.dump(metadict, jf)

# with h5py.File(os.path.join(save_folder, 'metadata.h5'), 'w') as h:
#     grp1 = h.create_group('hparms')
#     for key,val in hypers.items():
#         grp1.create_dataset(key, data = val)
#     grp2 = h.create_group('losses')
#     for key,val in loss.items():
#         grp2.create_dataset(key, data = val)
#     grp3 = h.create_group('misc')
#     grp3.create_dataset('duration', data = t_delta)


