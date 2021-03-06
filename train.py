### 06.12.2018
# Training a neural network on data processed in a way inspired by the kaggle tutorial
# This training file originally comes from the GitHub keras tutorial
import sys
import os
import time
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import h5py

sys.path.insert(1, 'C:\\Users\\Znerp\\Documents\\GitHub\\kaggle-dsb2-keras\\model')
from modelv01 import get_model
from git_utils import crps, real_to_cdf, preprocess, rotation_augmentation, shift_augmentation, load_train_data, split_data

data_folder = 'C:\\Users\\Znerp\\Documents\\GitHub\\kaggle-dsb2-keras\\data\\train'
save_folder = 'C:\\Users\\Znerp\\Documents\\GitHub\\kaggle-dsb2-keras\\training_results\\' + time.strftime('%Y%m%d_%H%M')

# create a new folder named after current date and time to save all model parameters and infos regarding the training process
os.makedirs(save_folder)

# Hyperparameters (implement in smart way)
hypers = {'model': 'modelv01',
'optimizer': 'Adam',
'lr_alpha': 1e-4, 
'nb_iter': 100, 
'epochs_per_iter': 1, 
'batch_size': 32, 
'calc_crps': 0}
# train_set_size = 1000 # first test with smaller subset


print('Loading and compiling models...')
model_systole = get_model()
model_diastole = get_model()
model_systole.summary()
time.sleep(1000)

# optimizer = Adam
# modelv01.compile(optimizer=hypers['optimizer'], loss=root_mean_squared_error)

print('Loading training data...')
filename = os.path.join(data_folder,'train_mri_64_64_N500.h5')
X, y = load_train_data(filename)
y = y.T

train_set_size = X.shape[0]
# test with smaller dataset
X = X[:train_set_size,:,:,:]
y = y[:train_set_size,:]

print('Pre-processing images...')
X = preprocess(X)

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

    print('Augmenting images - rotations')
    X_train_aug = rotation_augmentation(X_train, 15)
    print('Augmenting images - shifts')
    X_train_aug = shift_augmentation(X_train_aug, 0.1, 0.1)

    print('Fitting systole model...')
    hist_systole = model_systole.fit(X_train_aug, y_train[:, 0], batch_size=hypers['batch_size'], epochs=hypers['epochs_per_iter'], verbose = 2,
                                        validation_data=(X_test, y_test[:,0]), shuffle = True) 

    print('Fitting diastole model...')
    hist_diastole = model_diastole.fit(X_train_aug, y_train[:, 1], batch_size=hypers['batch_size'], epochs=hypers['epochs_per_iter'], verbose = 2,
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

# save losses and hyperparameters in additional h5py.file
with h5py.File(os.path.join(save_folder, 'metadata.h5'), 'w') as h:
    grp1 = h.create_group('hparms')
    for key,val in hypers.items():
        # print([key,val])
        grp1.create_dataset(key, data = val)
    grp2 = h.create_group('losses')
    for key,val in loss.items():
        grp2.create_dataset(key, data = val)
    grp3 = h.create_group('misc')
    grp3.create_dataset('duration', data = t_delta)


